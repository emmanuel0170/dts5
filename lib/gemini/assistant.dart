import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show ValueNotifier, debugPrint;

import '../audio/mic_client.dart';
import '../audio/voice_client.dart';
import '../decision/decision_engine.dart';
import '../decision/escalation.dart';
import '../decision/phrase_bank.dart';
import '../decision/speech_arbiter.dart';
import '../engine.dart';
import 'assistant_tools.dart';
import 'live_client.dart';
import 'live_protocol.dart';

enum AssistantState { off, connecting, listening, speaking }

class AssistantConfig {
  const AssistantConfig({
    this.model = 'gemini-3.8-live',
    this.voiceName,
    this.idleCloseMs = 25000,
    this.escalationIdleMs = 12000,
    this.closeAfterSearchMs = 4000,
    this.snapshotOnStart = true,
    this.snapshotAsClientTurn = true,
    this.halfDuplex = true,
    this.micTailMs = 600,
  });

  final String model;
  final String? voiceName; // a prebuilt voice name; null = the model's default
  final int idleCloseMs; // close the session after this much silence: on-demand, not always-on
  final int escalationIdleMs; // ...for a session that an escalation opened rather than the user
  final int closeAfterSearchMs; // ...once Gemini has started a search and finished its acknowledgement
  final bool snapshotOnStart; // send one camera frame as context when a conversation starts
  /// Snapshots from escalation go out as one explicit user turn (image + prompt), which is deterministic but
  /// interrupts any generation in progress. false = a realtime video frame followed by a text prompt.
  final bool snapshotAsClientTurn;

  /// Do not send microphone audio while Gemini is speaking (plus [micTailMs] after it stops). Without echo
  /// cancellation that works against it, the model hears its own voice (and the guidance tones), takes it for the
  /// user, interrupts itself and answers again: the "cut off and replayed" loop. The cost is that the user cannot
  /// talk over it by voice; the volume-up key interrupts instead.
  final bool halfDuplex;
  final int micTailMs;
}

/// On-demand Gemini Live conversation: opens when the user asks (or the decision engine escalates), streams the
/// microphone, plays Gemini's voice, runs its tool calls, and closes itself after a period of silence.
/// Guidance speech is held while Gemini talks; hazards still break through (see [interruptPlayback]).
class GeminiAssistant {
  GeminiAssistant({
    required this.transport,
    required this.auth,
    required this.decision,
    required this.speech,
    required this.voice,
    required this.mic,
    required this.snapshot,
    required this.vocabulary,
    this.config = const AssistantConfig(),
    this.onPlayingChanged,
  });

  final LiveTransport transport;

  /// Builds the WebSocket URI, including credentials. For development: the API key in the query string.
  /// For anything you ship: a short-lived token minted by your own server. Never ship a raw key.
  final Future<Uri> Function() auth;
  final DecisionEngine decision;
  final SpeechArbiter speech;
  final VoiceOut voice;
  final MicSource mic;
  final Future<Uint8List?> Function() snapshot;
  final List<String> vocabulary;
  final AssistantConfig config;

  /// Gemini started / stopped being audible (drives ducking of the sonification cues).
  final void Function(bool playing)? onPlayingChanged;

  final ValueNotifier<AssistantState> state = ValueNotifier(AssistantState.off);
  final ValueNotifier<String> lastUser = ValueNotifier('');
  final ValueNotifier<String> lastModel = ValueNotifier('');

  late final AssistantTools _tools = AssistantTools(
    decision: decision,
    vocabulary: vocabulary,
    sendCurrentView: _sendLookAround,
    onSearchStarted: () => _closeAfterSearch = true,
  );

  StreamSubscription<LiveEvent>? _events;
  StreamSubscription<Uint8List>? _micSub;
  Timer? _idle;
  Timer? _drain;
  String? _resumeHandle;
  bool _receiving = false; // audio of the current model turn is still arriving
  bool _playing = false;
  bool _dropTurnAudio = false; // a hazard cut this turn's audio off; ignore the rest of it
  bool _newTurn = true;
  bool _reconnecting = false;
  bool _closeAfterSearch = false; // a search was started from this conversation: close soon after Gemini finishes
  bool _escalationSession = false; // opened by an escalation rather than by the user
  int _micMutedUntilMs = 0;
  final Stopwatch _clock = Stopwatch()..start();

  /// Event counters shown on the Fused screen and printed to logcat (tag "[gemini ...]"): a diagnosis aid for
  /// "cut off and replayed" behaviour. Many `interrupted` events with no user speech point at echo or feedback;
  /// many `escalation` events at the escalation policy; `interruptPlayback` at urgent speech cutting Gemini off.
  final ValueNotifier<String> stats = ValueNotifier('');
  final Map<String, int> _counts = {};

  void _count(String what) {
    _counts[what] = (_counts[what] ?? 0) + 1;
    stats.value = _counts.entries.map((e) => '${e.key} ${e.value}').join('   ');
  }

  void _note(String what) {
    _count(what);
    debugPrint('[gemini ${_clock.elapsedMilliseconds}ms] $what');
  }

  bool get isActive => state.value != AssistantState.off;

  /// Start a conversation (the user asked). Returns false if it could not connect.
  Future<bool> startConversation() async {
    if (isActive) return true;
    if (!await _open()) return false;
    if (config.snapshotOnStart) await _sendLookAround();
    return true;
  }

  Future<void> stopConversation() => _teardown();

  /// The decision engine could not make sense of the scene: show Gemini a camera frame.
  Future<void> escalate(EscalationReason reason, FusedFrame frame) async {
    if (isActive) return; // the user is already in a conversation (or it is connecting): never pile on
    _note('escalation ${reason.name}');
    _escalationSession = true;
    if (!await _open()) {
      _escalationSession = false;
      return;
    }
    await _sendSnapshotTurn(_prompt(reason));
  }

  /// A hazard has to be spoken now: cut Gemini off and drop the rest of its current turn.
  void interruptPlayback() {
    if (!_playing && !_receiving) return;
    _note('interruptPlayback');
    _dropTurnAudio = true;
    _receiving = false;
    unawaited(voice.flush('gemini'));
    _ensureDrainWatch();
  }

  void dispose() {
    _events?.cancel();
    unawaited(_teardown());
  }

  // ------------------------------------------------------------------ session

  Map<String, dynamic> _setup() => buildSetup(
        model: config.model,
        systemInstruction: systemInstruction(vocabulary),
        functionDeclarations: toolDeclarations(vocabulary),
        voiceName: config.voiceName,
        resumptionHandle: _resumeHandle,
      );

  Future<bool> _open() async {
    state.value = AssistantState.connecting;
    _events ??= transport.events.listen(_onEvent, onError: (Object _) {});
    try {
      await transport.connect(await auth(), _setup());
    } catch (_) {
      await _teardown();
      speech.submit(const Utterance(['say:gemini_unavailable'],
          priority: Priority.status, key: 'gemini', ttlMs: 4000));
      return false;
    }
    try {
      await mic.start();
      _micSub = mic.chunks.listen(_onMic);
    } catch (_) {
      // No microphone (permission denied): the assistant can still answer escalations with its voice.
    }
    state.value = AssistantState.listening;
    _resetIdle();
    return true;
  }

  Future<void> _teardown() async {
    _idle?.cancel();
    _drain?.cancel();
    _drain = null;
    await _micSub?.cancel();
    _micSub = null;
    try {
      await mic.stop();
    } catch (_) {}
    if (transport.isOpen) transport.send(msgAudioStreamEnd());
    await transport.close();
    await voice.flush('gemini');
    _receiving = false;
    _dropTurnAudio = false;
    _resumeHandle = null;
    _reconnecting = false;
    _closeAfterSearch = false;
    _escalationSession = false;
    _micMutedUntilMs = 0;
    if (_playing) {
      _playing = false;
      speech.release();
      onPlayingChanged?.call(false);
    }
    state.value = AssistantState.off;
  }

  /// The server announced it will close the connection soon: reconnect with the resumption handle if we have one.
  Future<void> _reconnect() async {
    if (_reconnecting || !isActive || _resumeHandle == null) return;
    _reconnecting = true;
    try {
      await transport.connect(await auth(), _setup());
      _reconnecting = false;
    } catch (_) {
      await _teardown();
    }
  }

  // ------------------------------------------------------------------ input

  void _onMic(Uint8List chunk) {
    if (config.halfDuplex && (_playing || _clock.elapsedMilliseconds < _micMutedUntilMs)) return;
    transport.send(msgAudio(chunk));
  }

  Future<void> _sendLookAround() async {
    final jpeg = await snapshot();
    if (jpeg != null) transport.send(msgVideoJpeg(jpeg));
  }

  Future<void> _sendSnapshotTurn(String prompt) async {
    final jpeg = await snapshot();
    if (jpeg == null) return;
    _newTurn = true;
    if (config.snapshotAsClientTurn) {
      transport.send(msgClientTurn(text: prompt, jpeg: jpeg));
    } else {
      transport.send(msgVideoJpeg(jpeg));
      await Future<void>.delayed(const Duration(milliseconds: 200)); // let the frame land before the prompt
      transport.send(msgText(prompt));
    }
    _resetIdle();
  }

  String _prompt(EscalationReason r) {
    final goal = decision.guidance.goal;
    final looking = goal == null
        ? 'The user is not searching for anything.'
        : 'The user is looking for "${goal.phrase}".';
    final why = switch (r) {
      EscalationReason.searchStalled => 'The detector has not found it for a while.',
      EscalationReason.clutter => 'The scene is cluttered and the detector is unsure.',
      EscalationReason.depthUnreliable => 'The depth sensor is not giving useful readings here.',
    };
    return '[scene-check] $looking $why Using the attached camera image, say briefly whether you can see it '
        'and which way the user should turn.';
  }

  // ------------------------------------------------------------------ output

  void _onEvent(LiveEvent e) {
    switch (e) {
      case AudioChunk a:
        _onAudio(a);
      case InputTranscript t:
        _beginTurnIfNeeded();
        lastUser.value += t.text;
        _resetIdle();
      case OutputTranscript t:
        _beginTurnIfNeeded();
        lastModel.value += t.text;
      case ModelText t:
        _beginTurnIfNeeded();
        lastModel.value += t.text;
      case TurnComplete():
        _note('turnComplete');
        _receiving = false;
        _dropTurnAudio = false;
        _newTurn = true;
        _ensureDrainWatch();
      case Interrupted():
        _note('interrupted');
        _receiving = false;
        _dropTurnAudio = false;
        unawaited(voice.flush('gemini'));
        _ensureDrainWatch();
      case ToolCall c:
        _note('toolCall ' + c.calls.map((x) => x.name).join(', '));
        unawaited(_onToolCall(c));
      case ToolCallCancellation():
        break; // our tools are instant and idempotent
      case GoAway():
        _note('goAway');
        unawaited(_reconnect());
      case ResumptionUpdate u:
        if (u.resumable && u.handle != null) _resumeHandle = u.handle;
      case SocketClosed():
        _note('socketClosed');
        if (!_reconnecting) unawaited(_teardown());
      case SetupComplete():
        break;
    }
  }

  void _beginTurnIfNeeded() {
    if (!_newTurn) return;
    _newTurn = false;
    lastUser.value = '';
    lastModel.value = '';
  }

  void _onAudio(AudioChunk a) {
    _count('audioChunks');
    _beginTurnIfNeeded();
    if (_dropTurnAudio) return;
    _receiving = true;
    unawaited(voice.enqueue('gemini', a.pcm));
    if (!_playing) _setPlaying(true);
    _idle?.cancel();
    _ensureDrainWatch();
  }

  /// Gemini counts as "speaking" until its audio has actually finished playing, not just finished arriving.
  void _ensureDrainWatch() {
    _drain ??= Timer.periodic(const Duration(milliseconds: 120), (_) async {
      if (_receiving) return;
      if (await voice.pendingMs('gemini') > 0) return;
      _drain?.cancel();
      _drain = null;
      if (_playing) _setPlaying(false);
    });
  }

  void _setPlaying(bool playing) {
    _playing = playing;
    if (state.value == AssistantState.listening || state.value == AssistantState.speaking) {
      state.value = playing ? AssistantState.speaking : AssistantState.listening;
    }
    if (playing) {
      speech.hold();
    } else {
      _micMutedUntilMs = _clock.elapsedMilliseconds + config.micTailMs; // let the echo tail die away
      speech.release();
      _resetIdle();
    }
    onPlayingChanged?.call(playing);
  }

  Future<void> _onToolCall(ToolCall e) async {
    _resetIdle();
    final replies = <ToolReply>[];
    for (final c in e.calls) {
      try {
        replies.add(await _tools.handle(c));
      } catch (err) {
        replies.add(ToolReply(c.id, c.name, {'ok': false, 'error': '$err'}));
      }
    }
    transport.send(msgToolResponse(replies));
    // Started a search: wind the conversation down once Gemini has said its piece (the device guides from here).
    if (_closeAfterSearch) _resetIdle(overrideMs: 10000);
  }

  void _resetIdle({int? overrideMs}) {
    _idle?.cancel();
    if (!isActive) return;
    final ms = overrideMs ??
        (_closeAfterSearch
            ? config.closeAfterSearchMs
            : (_escalationSession ? config.escalationIdleMs : config.idleCloseMs));
    _idle = Timer(Duration(milliseconds: ms), () {
      if (_playing) {
        _resetIdle();
      } else {
        unawaited(_teardown());
      }
    });
  }
}
