import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'app_controller.dart';
import 'bench_screen.dart';
import 'engine.dart';
import 'gemini/assistant.dart';
import 'planning/planner.dart';
import 'screens/depth_screen.dart';
import 'screens/detector_screen.dart';
import 'screens/path_screen.dart';

/// Screen 3, the main screen: no camera view. The native engine owns the camera and both models; the controller
/// runs the decision engine (sonification, Kokoro phrases, Gemini). This screen starts/stops it and shows state.
class FusedScreen extends StatefulWidget {
  const FusedScreen({super.key, required this.controller});

  final AppController controller;

  @override
  State<FusedScreen> createState() => _FusedScreenState();
}

class _FusedScreenState extends State<FusedScreen> {
  AppController get _c => widget.controller;
  StreamSubscription<FusedFrame>? _sub;
  int _preset = 0;
  bool _running = false;
  String _status = 'Stopped';
  FusedFrame? _last;
  double _fps = 0;
  final _clock = Stopwatch()..start();
  int? _lastArrivalMs;

  @override
  void dispose() {
    _sub?.cancel();
    if (_running) _c.stop();
    super.dispose();
  }

  Future<void> _start() async {
    setState(() {
      _status = 'Starting (loading models takes a few seconds)...';
      _last = null;
      _fps = 0;
      _lastArrivalMs = null;
    });
    try {
      await _c.ensureInit();
      await _sub?.cancel();
      _sub = _c.frames.listen(_onFrame, onError: (Object e) {
        if (!mounted) return;
        setState(() {
          _status = 'ERROR: ${e is PlatformException ? e.message : e}';
          _running = false;
        });
      });
      await _c.start(EngineConfig.presets[_preset]);
      if (mounted) setState(() => _running = true);
    } on PlatformException catch (e) {
      if (mounted) setState(() => _status = 'ERROR ${e.code}: ${e.message}');
    } catch (e) {
      if (mounted) setState(() => _status = 'ERROR: $e');
    }
  }

  Future<void> _stop() async {
    await _c.stop();
    await _sub?.cancel();
    _sub = null;
    if (mounted) {
      setState(() {
        _running = false;
        _status = 'Stopped';
      });
    }
  }

  void _onFrame(FusedFrame f) {
    final now = _clock.elapsedMilliseconds;
    final prev = _lastArrivalMs;
    _lastArrivalMs = now;
    if (prev != null && now > prev) {
      final inst = 1000.0 / (now - prev);
      _fps = _fps == 0 ? inst : _fps * 0.9 + inst * 0.1;
    }
    if (!mounted) return;
    setState(() {
      _last = f;
      _status = 'Running';
    });
  }

  String _t(FusedFrame f, String key) => (f.timing[key] ?? double.nan).toStringAsFixed(0);

  String _m(double? v) => (v == null || v.isNaN) ? '  -  ' : '${v.toStringAsFixed(1)} m';

  @override
  Widget build(BuildContext context) {
    final f = _last;
    final grid = f?.grid;
    final dets = [...?f?.detections]..sort((a, b) {
        final da = (a.depthP10 == null || a.depthP10!.isNaN) ? double.infinity : a.depthP10!;
        final db = (b.depthP10 == null || b.depthP10!.isNaN) ? double.infinity : b.depthP10!;
        return da.compareTo(db);
      });

    return Scaffold(
      appBar: AppBar(
        title: const Text('Fused scene (headless)'),
        actions: [
          PopupMenuButton<int>(
            onSelected: (i) {
              // The views show the running engine, so they open on top of it. Only the benchmark needs the GPU free.
              final pages = <Widget>[
                DetectorScreen(controller: _c),
                DepthScreen(controller: _c),
                PathScreen(controller: _c),
                const BenchScreen(),
              ];
              if (_running && i == 3) _stop();
              Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => pages[i]));
            },
            itemBuilder: (_) => const [
              PopupMenuItem(value: 0, child: Text('1 · Detector (YOLOE)')),
              PopupMenuItem(value: 1, child: Text('2 · Depth')),
              PopupMenuItem(value: 2, child: Text('3 · Path planning')),
              PopupMenuItem(value: 3, child: Text('Benchmark')),
            ],
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              DropdownButton<int>(
                isExpanded: true,
                value: _preset,
                onChanged: _running ? null : (v) => setState(() => _preset = v ?? 0),
                items: [
                  for (var i = 0; i < EngineConfig.presets.length; i++)
                    DropdownMenuItem(value: i, child: Text(EngineConfig.presets[i].name)),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: FilledButton(
                      onPressed: _running ? _stop : _start,
                      child: Text(_running ? 'Stop' : 'Start'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(child: _talkButton()),
                ],
              ),
              const SizedBox(height: 8),
              Text(_status),
              if (_running)
                const Text('Volume UP: talk to Gemini (or interrupt it)   Volume DOWN: stop',
                    style: TextStyle(fontSize: 12)),
              if (f != null)
                Text(
                  '${_fps.toStringAsFixed(1)} fps   frame ${f.frameId}   ${f.w}x${f.h}\n'
                  'total ${_t(f, "totalMs")} ms (convert ${_t(f, "convMs")}, models ${_t(f, "modelsMs")}, '
                  'fuse ${_t(f, "fuseMs")})\n'
                  'depth ${f.depthFresh ? "fresh" : "reused"}, age ${f.depthAgeMs.toStringAsFixed(0)} ms\n'
                  'yolo: pre ${_t(f, "yPreMs")} enq ${_t(f, "yRunMs")} wait ${_t(f, "yReadMs")} dec ${_t(f, "decodeMs")}\n'
                  'depth: pre+enq ${_t(f, "dEnqMs")} wait ${_t(f, "dReadMs")}',
                  style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                ),
              if (_running) _decisionPanel(),
              const Divider(),
              Expanded(
                child: ListView(
                  children: [
                    for (final d in dets)
                      Text(
                        '${d.label.padRight(14)} ${d.score.toStringAsFixed(2)}  '
                        'median ${_m(d.depthMedian)}  near ${_m(d.depthP10)}',
                        style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                      ),
                    if (f != null && dets.isEmpty)
                      const Text('no detections', style: TextStyle(fontStyle: FontStyle.italic)),
                    const SizedBox(height: 12),
                    if (f != null && grid != null) ...[
                      const Text('depth grid (nearest 10% per cell, metres; top of frame first)'),
                      for (var r = 0; r < f.gridRows; r++)
                        Text(
                          [
                            for (var c = 0; c < f.gridCols; c++)
                              grid[r * f.gridCols + c].toStringAsFixed(1).padLeft(6),
                          ].join(' '),
                          style: const TextStyle(fontFamily: 'monospace', fontSize: 14),
                        ),
                    ],
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _talkButton() {
    if (!_c.geminiAvailable) {
      return const Tooltip(
        message: 'Run with --dart-define=GEMINI_API_KEY=...',
        child: FilledButton.tonal(onPressed: null, child: Text('Gemini: no key')),
      );
    }
    return FutureBuilder<void>(
      future: _c.ensureInit(),
      builder: (context, snap) {
        if (snap.hasError) {
          return const FilledButton.tonal(onPressed: null, child: Text('Setup error'));
        }
        if (snap.connectionState != ConnectionState.done) {
          return const FilledButton.tonal(onPressed: null, child: Text('Talk'));
        }
        return ValueListenableBuilder<AssistantState>(
          valueListenable: _c.assistant.state,
          builder: (context, s, _) => FilledButton.tonal(
            onPressed: _running ? _c.toggleConversation : null,
            child: Text(switch (s) {
              AssistantState.off => 'Talk to Gemini',
              AssistantState.connecting => 'Connecting...',
              AssistantState.listening => 'Listening (tap to end)',
              AssistantState.speaking => 'Gemini speaking',
            }),
          ),
        );
      },
    );
  }

  /// What the decision engine is doing, and the Gemini transcript.
  Widget _decisionPanel() {
    final g = _c.decision.guidance;
    final goal = g.goal;
    return Padding(
      padding: const EdgeInsets.only(top: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'search: ${goal == null ? "none" : "${goal.labels.join("/")}  [${g.phase.name}]"}'
            '   phrases loaded: ${_c.clips.count}',
            style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
          ),
          ValueListenableBuilder<PlanResult?>(
            valueListenable: _c.decision.plan,
            builder: (_, p, __) => Text(
              p == null
                  ? 'plan: waiting for depth'
                  : !p.ok
                      ? 'plan: no ground plane (coarse grid fallback)'
                      : 'plan: ${p.status.name}  steer ${p.steerDeg?.toStringAsFixed(0) ?? "-"} deg  '
                          'L ${p.left?.rangeM.toStringAsFixed(1) ?? "-"}  C ${p.center?.rangeM.toStringAsFixed(1) ?? "-"}  '
                          'R ${p.right?.rangeM.toStringAsFixed(1) ?? "-"} m',
              style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
            ),
          ),
          ValueListenableBuilder<String>(
            valueListenable: _c.assistant.stats,
            builder: (_, v, __) => v.isEmpty
                ? const SizedBox.shrink()
                : Text('gemini events: $v', style: const TextStyle(fontFamily: 'monospace', fontSize: 11)),
          ),
          ValueListenableBuilder<String>(
            valueListenable: _c.assistant.lastUser,
            builder: (_, v, __) => v.isEmpty ? const SizedBox.shrink() : Text('you: $v'),
          ),
          ValueListenableBuilder<String>(
            valueListenable: _c.assistant.lastModel,
            builder: (_, v, __) => v.isEmpty ? const SizedBox.shrink() : Text('gemini: $v'),
          ),
        ],
      ),
    );
  }
}
