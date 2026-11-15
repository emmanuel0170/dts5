import 'dart:async';

import 'package:flutter/foundation.dart' show ValueNotifier;
import 'package:flutter/services.dart' show HapticFeedback, rootBundle;

import 'audio/mic_client.dart';
import 'audio/phrase_clips.dart';
import 'audio/voice_client.dart';
import 'decision/cue_player.dart';
import 'decision/decision_engine.dart';
import 'decision/speech_arbiter.dart';
import 'engine.dart';
import 'gemini/assistant.dart';
import 'gemini/live_client.dart';

/// Wires everything together: camera/model engine -> decision engine -> {sonification cues, Kokoro clips,
/// Gemini}. One instance per app.
///
/// Hardware volume keys (while the engine runs) are the accessible control for Gemini, so nothing needs to be
/// tapped on the screen:  volume UP  = start a conversation (or, while Gemini is speaking, interrupt it);
/// volume DOWN = end the conversation (or, with no conversation, cancel the current search).
/// Each press gives a haptic tick and a short rising / falling confirmation tone.
class AppController {
  AppController({this.geminiApiKey = const String.fromEnvironment('GEMINI_API_KEY')});

  /// Development only (`flutter run --dart-define=GEMINI_API_KEY=...`). A key compiled into an app can be
  /// extracted from it; for anything you distribute, mint short-lived tokens on your own server instead.
  final String geminiApiKey;

  final engine = SceneEngine();
  final cues = CuePlayer();
  final voice = VoiceClient();
  final mic = MicClient();

  late final SpeechArbiter speech;
  late final DecisionEngine decision;
  late final GeminiAssistant assistant;
  late final PhraseClips clips;
  late final List<String> vocabulary;

  /// true while the engine (camera + models + decision loop) is running.
  final running = ValueNotifier<bool>(false);

  final _frames = StreamController<FusedFrame>.broadcast();
  StreamSubscription<FusedFrame>? _sub;
  StreamSubscription<String>? _keySub;
  Future<void>? _initFuture;
  bool _ready = false;
  bool _speechSpeaking = false;

  bool get geminiAvailable => geminiApiKey.isNotEmpty;

  /// One fused frame per camera frame, for the UI. (The decision engine is fed internally.)
  Stream<FusedFrame> get frames => _frames.stream;

  Future<void> ensureInit() => _initFuture ??= _init();

  Future<void> _init() async {
    // assets/labels.txt must be a copy of the labels.txt the detector was exported with (same order).
    vocabulary = (await rootBundle.loadString('assets/labels.txt'))
        .split('\n')
        .map((l) => l.trim())
        .where((l) => l.isNotEmpty)
        .toList();
    clips = await PhraseClips.load();

    speech = SpeechArbiter(
      ClipSpeechOutput(clips, voice),
      onPreemptWhileHeld: () => assistant.interruptPlayback(),
      onSpeakingChanged: (s) {
        _speechSpeaking = s;
        _updateDuck();
      },
    );
    decision = DecisionEngine(speech: speech);
    decision.onCues = cues.update;
    decision.onEscalate = (reason, frame) => assistant.escalate(reason, frame);
    decision.canEscalate = () => !assistant.isActive;

    assistant = GeminiAssistant(
      transport: WebSocketLiveTransport(),
      auth: () async => Uri.parse(
        'wss://generativelanguage.googleapis.com/ws/'
        'google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=$geminiApiKey',
      ),
      decision: decision,
      speech: speech,
      voice: voice,
      mic: mic,
      snapshot: engine.snapshot,
      vocabulary: vocabulary,
    );
    assistant.state.addListener(_updateDuck);
    _ready = true;
  }

  /// While a conversation is open the beacon is silenced entirely (the microphone would pick up its pulses and
  /// Gemini would take them for the user); hazard tones stay. While a guidance phrase plays it is turned down.
  void _updateDuck() {
    final level = assistant.isActive ? 0.0 : (_speechSpeaking ? 0.3 : 1.0);
    cues.setDuckLevel(level);
  }

  Future<void> start(EngineConfig config) async {
    await ensureInit();
    await cues.start();
    await voice.start('guidance');
    await voice.start('gemini');
    await _sub?.cancel();
    _sub = engine.frames().listen(
      (f) {
        decision.onFrame(f);
        _frames.add(f);
      },
      onError: _frames.addError,
    );
    await engine.start(config);
    await engine.setCaptureVolumeKeys(true);
    await _keySub?.cancel();
    _keySub = engine.volumeKeys.listen(_onVolumeKey);
    running.value = true;
  }

  Future<void> stop() async {
    running.value = false;
    await _keySub?.cancel();
    _keySub = null;
    if (_ready) {
      await assistant.stopConversation();
      decision.clearGoal();
    }
    try {
      await engine.setCaptureVolumeKeys(false); // give the volume keys back
    } catch (_) {}
    await engine.stop();
    await _sub?.cancel();
    _sub = null;
    await cues.stop();
    await voice.stop('guidance');
    await voice.stop('gemini');
  }

  /// Visualisation screens turn this on while visible; every frame then carries a camera preview.
  Future<void> setViz(bool on) => engine.setViz(on);

  Future<void> _onVolumeKey(String key) async {
    if (!_ready) return;
    HapticFeedback.mediumImpact();
    if (key == 'up') {
      if (!geminiAvailable) {
        cues.earcon('alert'); // no key configured: say so with a sound
        return;
      }
      if (!assistant.isActive) {
        cues.earcon('listen_on');
        await assistant.startConversation();
      } else if (assistant.state.value == AssistantState.speaking) {
        assistant.interruptPlayback(); // barge in: stop talking, I want to speak
      }
    } else {
      if (assistant.isActive) {
        await assistant.stopConversation();
        cues.earcon('listen_off');
      } else if (decision.guidance.goal != null) {
        decision.clearGoal();
        cues.earcon('listen_off');
      }
    }
  }

  /// On-screen equivalent of the volume keys (for development).
  Future<void> toggleConversation() async {
    await ensureInit();
    if (!geminiAvailable) return;
    if (assistant.isActive) {
      await assistant.stopConversation();
    } else {
      await assistant.startConversation();
    }
  }
}
