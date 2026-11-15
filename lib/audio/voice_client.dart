import 'dart:typed_data';

import 'package:flutter/services.dart';

/// Native PCM players: 'guidance' (pre-rendered Kokoro clips) and 'gemini' (Gemini's streamed voice).
/// Both play 16-bit mono PCM at 24 kHz.
abstract class VoiceOut {
  /// With [wait], completes when the audio has played (or was flushed); otherwise returns immediately.
  Future<void> enqueue(String voice, Uint8List pcm, {bool wait = false});
  Future<void> flush(String voice);
  Future<int> pendingMs(String voice);
}

class VoiceClient implements VoiceOut {
  static const _ch = MethodChannel('spatial_nav/voice');

  Future<void> start(String voice) => _ch.invokeMethod<void>('start', {'voice': voice});
  Future<void> stop(String voice) => _ch.invokeMethod<void>('stop', {'voice': voice});

  @override
  Future<void> enqueue(String voice, Uint8List pcm, {bool wait = false}) =>
      _ch.invokeMethod<void>('enqueue', {'voice': voice, 'pcm': pcm, 'await': wait});

  @override
  Future<void> flush(String voice) => _ch.invokeMethod<void>('flush', {'voice': voice});

  @override
  Future<int> pendingMs(String voice) async =>
      (await _ch.invokeMethod<int>('pendingMs', {'voice': voice})) ?? 0;
}
