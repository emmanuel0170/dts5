import 'dart:async';

import 'package:flutter/services.dart';

import 'cue_mapper.dart';

/// Dart side of the native cue synth (CueSynth.kt). Fire-and-forget: the synth holds the last parameters, keeps
/// its own pulse timing, and fades out by itself if updates stop for ~0.7 s.
class CuePlayer {
  static const _channel = MethodChannel('spatial_nav/cues');

  Future<void> start() => _channel.invokeMethod<void>('start');
  Future<void> stop() => _channel.invokeMethod<void>('stop');

  void update(CueFrame frame) => unawaited(_channel.invokeMethod<void>('update', frame.toMap()));

  /// A one-shot sound without touching the continuous streams (listen_on, listen_off, found, ...).
  void earcon(String name) => unawaited(_channel.invokeMethod<void>('earcon', name));

  /// 1 = normal, 0 = silent: applied to the beacon and the earcons. Hazard tones are never ducked.
  void setDuckLevel(double level) => unawaited(_channel.invokeMethod<void>('duck', level));
}
