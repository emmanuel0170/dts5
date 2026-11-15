import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../app_controller.dart';
import '../planning/planner.dart';
import 'viz_frame.dart';

/// Screen 2: the depth map as the planner receives it (48 x 64 metric depth, median of 3x3 per cell), coloured from
/// red (near) to blue (far) over 0.3-8 m, optionally laid over the camera image.
class DepthScreen extends StatefulWidget {
  const DepthScreen({super.key, required this.controller});
  final AppController controller;

  @override
  State<DepthScreen> createState() => _DepthScreenState();
}

class _DepthScreenState extends State<DepthScreen> {
  static const double _near = 0.3, _far = 8.0;
  static final List<Color> _lut = [
    for (var i = 0; i < 256; i++) HSVColor.fromAHSV(1, 240.0 * i / 255, 1, 1).toColor(),
  ];

  ui.Image? _image;
  int _renderedSerial = -1;
  double _overlay = 0.0; // 0 = depth only, 1 = camera image only

  void _render(Float32List grid) {
    final rgba = Uint8List(kGridW * kGridH * 4);
    for (var k = 0; k < grid.length; k++) {
      final d = grid[k];
      final o = k * 4;
      if (d.isNaN) {
        rgba[o] = 40;
        rgba[o + 1] = 40;
        rgba[o + 2] = 40;
      } else {
        final t = ((d - _near) / (_far - _near)).clamp(0.0, 1.0);
        final c = _lut[(t * 255).round()];
        rgba[o] = c.red;
        rgba[o + 1] = c.green;
        rgba[o + 2] = c.blue;
      }
      rgba[o + 3] = 255;
    }
    ui.decodeImageFromPixels(rgba, kGridW, kGridH, ui.PixelFormat.rgba8888, (img) {
      if (mounted) setState(() => _image = img);
    });
  }

  @override
  Widget build(BuildContext context) => VizFrame(
        controller: widget.controller,
        title: '2 · Depth',
        builder: (context, d) {
          if (d.depthGrid != null && d.depthSerial != _renderedSerial) {
            _renderedSerial = d.depthSerial;
            _render(d.depthGrid!);
          }
          final plan = widget.controller.decision.plan.value;
          return Column(
            children: [
              AspectRatio(
                aspectRatio: 3 / 4,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    if (_image != null)
                      RawImage(image: _image, fit: BoxFit.fill, filterQuality: FilterQuality.medium)
                    else
                      const ColoredBox(color: Colors.black),
                    if (_overlay > 0) PreviewImage(d.preview, opacity: _overlay),
                  ],
                ),
              ),
              Row(
                children: [
                  const SizedBox(width: 8),
                  const Text('depth', style: TextStyle(fontSize: 12)),
                  Expanded(
                    child: Slider(value: _overlay, onChanged: (v) => setState(() => _overlay = v)),
                  ),
                  const Text('camera', style: TextStyle(fontSize: 12)),
                  const SizedBox(width: 8),
                ],
              ),
              Container(
                height: 14,
                margin: const EdgeInsets.symmetric(horizontal: 12),
                decoration: BoxDecoration(gradient: LinearGradient(colors: _lut)),
              ),
              const Padding(
                padding: EdgeInsets.symmetric(horizontal: 12),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [Text('0.3 m'), Text('~4 m'), Text('8 m+'), Text('grey = no depth')],
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(8),
                child: Text(
                  plan == null || !plan.ok
                      ? 'ground plane: not found (needs a stretch of visible floor)'
                      : 'ground plane: camera height ${plan.heightM.toStringAsFixed(2)} m, pitched down '
                          '${plan.pitchDeg.toStringAsFixed(0)} deg${plan.held ? " (held)" : ""}',
                  style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                ),
              ),
            ],
          );
        },
      );
}
