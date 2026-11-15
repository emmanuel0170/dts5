import 'package:flutter/material.dart';

import '../app_controller.dart';
import '../engine.dart';
import 'viz_frame.dart';

/// Screen 1: what the YOLOE detector sees. The camera preview with every detection's box, label, confidence and the
/// distance read from the depth map.
class DetectorScreen extends StatelessWidget {
  const DetectorScreen({super.key, required this.controller});
  final AppController controller;

  @override
  Widget build(BuildContext context) => VizFrame(
        controller: controller,
        title: '1 · Detector (YOLOE)',
        builder: (context, d) {
          final dets = d.frame?.detections ?? const <DetectionObj>[];
          return Column(
            children: [
              AspectRatio(
                aspectRatio: 3 / 4,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    PreviewImage(d.preview),
                    CustomPaint(painter: _BoxPainter(dets)),
                  ],
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(8),
                child: Text(
                  '${dets.length} detections   frame ${d.frame?.frameId ?? "-"}   '
                  'total ${(d.frame?.timing["totalMs"] ?? double.nan).toStringAsFixed(0)} ms',
                  style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                ),
              ),
              Expanded(
                child: ListView(
                  padding: const EdgeInsets.symmetric(horizontal: 8),
                  children: [
                    for (final o in dets)
                      Text(
                        '${o.label.padRight(14)} ${o.score.toStringAsFixed(2)}  '
                        '${o.depthMedian == null || o.depthMedian!.isNaN ? "   -  " : "${o.depthMedian!.toStringAsFixed(1)} m"}',
                        style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                      ),
                  ],
                ),
              ),
            ],
          );
        },
      );
}

class _BoxPainter extends CustomPainter {
  _BoxPainter(this.dets);
  final List<DetectionObj> dets;

  static Color _color(String label) {
    final h = label.codeUnits.fold<int>(7, (a, c) => (a * 31 + c) & 0xFFFF);
    return HSVColor.fromAHSV(1, (h % 360).toDouble(), 0.85, 1).toColor();
  }

  @override
  void paint(Canvas canvas, Size size) {
    for (final o in dets) {
      final color = _color(o.label);
      final r = Rect.fromLTRB(o.box[0] * size.width, o.box[1] * size.height, o.box[2] * size.width, o.box[3] * size.height);
      canvas.drawRect(r, Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2
        ..color = color);
      final dist = (o.depthMedian == null || o.depthMedian!.isNaN) ? '' : ' ${o.depthMedian!.toStringAsFixed(1)}m';
      final tp = TextPainter(
        text: TextSpan(
          text: '${o.label} ${o.score.toStringAsFixed(2)}$dist',
          style: const TextStyle(color: Colors.black, fontSize: 11, fontWeight: FontWeight.w600),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      final bg = Rect.fromLTWH(r.left, (r.top - tp.height).clamp(0.0, size.height), tp.width + 6, tp.height);
      canvas.drawRect(bg, Paint()..color = color);
      tp.paint(canvas, Offset(bg.left + 3, bg.top));
    }
  }

  @override
  bool shouldRepaint(covariant _BoxPainter old) => true;
}
