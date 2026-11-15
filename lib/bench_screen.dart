import 'package:flutter/material.dart';
import 'package:flutter/services.dart';


class BenchScreen extends StatefulWidget {
  const BenchScreen({super.key});

  @override
  State<BenchScreen> createState() => _BenchScreenState();
}

class _BenchScreenState extends State<BenchScreen> {
  static const _channel = MethodChannel('spatial_nav/bench');

  String _report = 'Tap Run. Keep the app in the foreground; this takes about a minute.';
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _showLast();
  }

  /// If a previous run died natively (e.g. a GPU backend crash), its partial report is still on disk.
  Future<void> _showLast() async {
    try {
      final last = await _channel.invokeMethod<String>('last');
      if (last != null && last.isNotEmpty && mounted) {
        setState(() => _report = 'LAST RUN (may be partial):\n\n$last');
      }
    } on PlatformException {
      // ignore: nothing to show
    }
  }

  Future<void> _run() async {
    setState(() {
      _busy = true;
      _report = 'Running...';
    });
    try {
      final r = await _channel.invokeMethod<String>('run');
      setState(() => _report = r ?? 'no result');
    } on PlatformException catch (e) {
      setState(() => _report = 'ERROR ${e.code}: ${e.message}');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('On-device benchmark')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(
                children: [
                  Expanded(
                    child: FilledButton(
                      onPressed: _busy ? null : _run,
                      child: Text(_busy ? 'Running...' : 'Run benchmark'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton(
                    onPressed: () => Clipboard.setData(ClipboardData(text: _report)),
                    child: const Text('Copy'),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Expanded(
                child: SingleChildScrollView(
                  child: SelectableText(
                    _report,
                    style: const TextStyle(fontFamily: 'monospace', fontSize: 11.5),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
