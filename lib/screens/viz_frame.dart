import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../app_controller.dart';
import '../engine.dart';

/// The latest of each kind of data. Depth grids arrive only on frames with a fresh depth map and previews only
/// while a screen has asked for them, so the screens keep the most recent one of each.
class VizData {
  FusedFrame? frame;
  Float32List? depthGrid;
  Uint8List? preview;
  double? hfovDeg;
  int depthSerial = 0; // increments with every new depth grid
}

/// Shared shell for the visualisation screens: shows the data of the running engine, switches the camera preview
/// on while visible and off again afterwards, and explains itself if the engine is not running.
class VizFrame extends StatefulWidget {
  const VizFrame({
    super.key,
    required this.controller,
    required this.title,
    required this.builder,
    this.needsPreview = true,
  });

  final AppController controller;
  final String title;
  final bool needsPreview;
  final Widget Function(BuildContext context, VizData data) builder;

  @override
  State<VizFrame> createState() => _VizFrameState();
}

class _VizFrameState extends State<VizFrame> {
  final _data = VizData();
  StreamSubscription<FusedFrame>? _sub;

  @override
  void initState() {
    super.initState();
    if (widget.needsPreview) widget.controller.setViz(true);
    _sub = widget.controller.frames.listen((f) {
      if (!mounted) return;
      setState(() {
        _data.frame = f;
        _data.hfovDeg = f.hfovDeg ?? _data.hfovDeg;
        if (f.depthGrid != null) {
          _data.depthGrid = f.depthGrid;
          _data.depthSerial++;
        }
        if (f.previewJpeg != null) _data.preview = f.previewJpeg;
      });
    });
  }

  @override
  void dispose() {
    _sub?.cancel();
    if (widget.needsPreview) widget.controller.setViz(false);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(widget.title)),
      body: SafeArea(
        child: ValueListenableBuilder<bool>(
          valueListenable: widget.controller.running,
          builder: (context, running, _) {
            if (!running) {
              return const Center(
                child: Padding(
                  padding: EdgeInsets.all(24),
                  child: Text('The engine is not running. Press Start on the main screen, then open this view '
                      'from the menu.'),
                ),
              );
            }
            return widget.builder(context, _data);
          },
        ),
      ),
    );
  }
}

/// Camera preview scaled to the 3:4 portrait frame, or a black box until the first one arrives.
class PreviewImage extends StatelessWidget {
  const PreviewImage(this.jpeg, {super.key, this.opacity = 1.0});
  final Uint8List? jpeg;
  final double opacity;

  @override
  Widget build(BuildContext context) {
    final j = jpeg;
    if (j == null) return const ColoredBox(color: Colors.black);
    return Opacity(opacity: opacity, child: Image.memory(j, gaplessPlayback: true, fit: BoxFit.fill));
  }
}
