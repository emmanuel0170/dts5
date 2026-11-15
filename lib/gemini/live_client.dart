import 'dart:async';
import 'dart:convert';

import 'package:web_socket_channel/web_socket_channel.dart';

import 'live_protocol.dart';

/// The connection to the Live API. An interface so the assistant can be tested with a fake.
abstract class LiveTransport {
  Stream<LiveEvent> get events;
  bool get isOpen;

  /// Opens the socket, sends [setup], and completes once the server answers setupComplete.
  Future<void> connect(Uri uri, Map<String, dynamic> setup);

  void send(Map<String, dynamic> message);
  Future<void> close();
}

class WebSocketLiveTransport implements LiveTransport {
  final _events = StreamController<LiveEvent>.broadcast();
  WebSocketChannel? _channel;
  StreamSubscription<dynamic>? _sub;
  bool _open = false;
  bool _closingByUs = false;

  @override
  Stream<LiveEvent> get events => _events.stream;

  @override
  bool get isOpen => _open;

  @override
  Future<void> connect(Uri uri, Map<String, dynamic> setup) async {
    await close(); // a reconnect (session resumption) reuses this object
    _closingByUs = false;
    final ch = WebSocketChannel.connect(uri);
    _channel = ch;
    await ch.ready; // throws if the connection could not be made

    final setupDone = Completer<void>();
    _sub = ch.stream.listen(
      (raw) {
        try {
          for (final e in parseServerMessage(raw as Object)) {
            if (e is SetupComplete && !setupDone.isCompleted) setupDone.complete();
            _events.add(e);
          }
        } catch (err) {
          _events.addError(err);
        }
      },
      onError: (Object err) {
        _open = false;
        if (!setupDone.isCompleted) setupDone.completeError(err);
        _events.addError(err);
      },
      onDone: () {
        _open = false;
        if (!setupDone.isCompleted) setupDone.completeError(StateError('socket closed before setup completed'));
        if (!_closingByUs) _events.add(SocketClosed(ch.closeCode, ch.closeReason));
      },
    );
    _open = true;
    ch.sink.add(jsonEncode(setup));
    await setupDone.future.timeout(const Duration(seconds: 10));
  }

  @override
  void send(Map<String, dynamic> message) {
    if (_open) _channel?.sink.add(jsonEncode(message));
  }

  @override
  Future<void> close() async {
    _closingByUs = true;
    _open = false;
    final sub = _sub;
    final ch = _channel;
    _sub = null;
    _channel = null;
    await sub?.cancel();
    await ch?.sink.close();
  }
}
