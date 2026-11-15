import 'dart:typed_data';

import 'package:flutter/services.dart';

/// 16 kHz mono PCM16 microphone chunks (40 ms each). An interface so the assistant can be tested with a fake.
abstract class MicSource {
  Stream<Uint8List> get chunks;
  Future<void> start();
  Future<void> stop();
}

class MicClient implements MicSource {
  static const _method = MethodChannel('spatial_nav/mic');
  static const _events = EventChannel('spatial_nav/mic_events');

  @override
  Stream<Uint8List> get chunks => _events.receiveBroadcastStream().map((e) => e as Uint8List);

  /// Asks for the microphone permission if needed; throws a PlatformException if denied.
  @override
  Future<void> start() => _method.invokeMethod<void>('start');

  @override
  Future<void> stop() => _method.invokeMethod<void>('stop');
}
