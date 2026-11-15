import 'phrase_bank.dart';

/// Plays a sequence of phrase-bank clips. [speak] completes when playback ends or [stop] is called.
abstract class SpeechOutput {
  Future<void> speak(List<String> parts);
  Future<void> stop();
}

class _Queued {
  _Queued(this.u, this.queuedAt);
  final Utterance u;
  final double queuedAt;
}

/// One voice at a time. Priority order, replace-by-key, expiry of stale messages, and hold/release while
/// Gemini (or the user) is speaking. Only urgent utterances break through a hold.
class SpeechArbiter {
  SpeechArbiter(this._out,
      {double Function()? nowMs, this.onPreemptWhileHeld, this.onSpeakingChanged, this.preemptCooldownMs = 8000})
      : _now = nowMs ?? (() => DateTime.now().millisecondsSinceEpoch.toDouble());

  final SpeechOutput _out;
  final double Function() _now;

  /// Called when a hazard has to be spoken while held, so the caller can stop Gemini's playback.
  final void Function()? onPreemptWhileHeld;

  /// An urgent utterance may cut Gemini off at most this often; during the cool-down it is dropped.
  final double preemptCooldownMs;
  double _lastPreemptMs = -1e12;

  /// true when the first clip of a burst starts, false when the queue has drained (use it to duck the sonification).
  final void Function(bool speaking)? onSpeakingChanged;

  final List<_Queued> _queue = [];
  bool _speaking = false;
  Priority? _current;
  bool _held = false;

  bool get isSpeaking => _speaking;

  void submit(Utterance u) {
    final t = _now();
    if (_held && u.urgent) {
      if (t - _lastPreemptMs < preemptCooldownMs) return; // do not keep cutting the conversation off
      _lastPreemptMs = t;
      onPreemptWhileHeld?.call();
    }
    if (u.key != null) _queue.removeWhere((q) => q.u.key == u.key);
    _queue.add(_Queued(u, t));
    if (u.priority == Priority.hazard && _speaking && _current != Priority.hazard) _out.stop();
    _pump();
  }

  /// Gemini or the user is speaking: pause guidance and stop whatever is playing (unless it is a hazard).
  void hold() {
    _held = true;
    if (_speaking && _current != Priority.hazard) _out.stop();
  }

  void release() {
    _held = false;
    _pump();
  }

  void clear() => _queue.clear();

  Future<void> _pump() async {
    if (_speaking) return;
    var announced = false;
    while (true) {
      final t = _now();
      _queue.removeWhere((q) => t - q.queuedAt > q.u.ttlMs);
      if (_queue.isEmpty) break;
      final allowed = _held ? _queue.where((q) => q.u.urgent).toList() : _queue;
      if (allowed.isEmpty) break;
      allowed.sort((a, b) {
        final c = b.u.priority.index.compareTo(a.u.priority.index);
        return c != 0 ? c : a.queuedAt.compareTo(b.queuedAt);
      });
      final next = allowed.first;
      _queue.remove(next);
      _speaking = true;
      _current = next.u.priority;
      if (!announced) {
        announced = true;
        onSpeakingChanged?.call(true);
      }
      try {
        await _out.speak(next.u.parts);
      } finally {
        _speaking = false;
        _current = null;
      }
    }
    if (announced) onSpeakingChanged?.call(false);
  }
}
