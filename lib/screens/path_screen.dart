import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../app_controller.dart';
import '../planning/planner.dart';
import 'viz_frame.dart';

/// Screen 3-viz: what the path planner makes of the scene, from above. The user stands at the bottom centre looking
/// up. Green = floor seen, red = obstacle, purple = drop, dark = unknown; amber = obstacle margin (the body is
/// treated as 0.8 m wide). The cyan line is the planned path, the white arrow the steering direction, the gold
/// diamond the goal (the target if one is being guided to, otherwise a point 4 m ahead).
class PathScreen extends StatelessWidget {
  const PathScreen({super.key, required this.controller});
  final AppController controller;

  @override
  Widget build(BuildContext context) => VizFrame(
        controller: controller,
        title: 'Path planning',
        builder: (context, d) {
          final plan = controller.decision.plan.value;
          final hfov = d.hfovDeg ?? 60.0;
          return Column(
            children: [
              AspectRatio(
                aspectRatio: 1,
                child: Stack(
                  children: [
                    Positioned.fill(child: CustomPaint(painter: _BevPainter(plan, hfov))),
                    // small camera thumbnail for orientation
                    Positioned(
                      right: 6,
                      top: 6,
                      width: 72,
                      height: 96,
                      child: DecoratedBox(
                        decoration: BoxDecoration(border: Border.all(color: Colors.white54)),
                        child: PreviewImage(d.preview),
                      ),
                    ),
                  ],
                ),
              ),
              Expanded(child: SingleChildScrollView(padding: const EdgeInsets.all(8), child: _info(plan))),
            ],
          );
        },
      );

  Widget _info(PlanResult? p) {
    String s(double? v, [String unit = ' m']) => v == null ? '-' : '${v.toStringAsFixed(1)}$unit';
    String hit(SectorHit? h) => h == null ? '-' : '${h.rangeM.toStringAsFixed(1)} m @ ${h.azimuthDeg.toStringAsFixed(0)} deg';
    final lines = p == null
        ? ['waiting for a depth map...']
        : !p.ok
            ? ['status: no ground plane', 'The floor is not visible enough to plan. Obstacle tones fall back to the coarse depth grid.']
            : [
                'status: ${p.status.name}${p.held ? " (plane held)" : ""}   goal ${p.hasGoal ? "target" : "straight ahead"}'
                    '${p.reached ? "" : " (not reachable: closest approach)"}',
                'camera ${p.heightM.toStringAsFixed(2)} m above floor, pitched down ${p.pitchDeg.toStringAsFixed(0)} deg',
                'steer: ${s(p.steerDeg, " deg")}   open bearing: ${s(p.openBearingDeg, " deg")} (${s(p.openRunM)} free)',
                'obstacle straight ahead: ${s(p.obstacleAheadM)}   free ahead to: ${s(p.freeAheadM)}',
                'left: ${hit(p.left)}',
                'centre: ${hit(p.center)}',
                'right: ${hit(p.right)}',
              ];
    return Text(lines.join('\n'), style: const TextStyle(fontFamily: 'monospace', fontSize: 12));
  }
}

class _BevPainter extends CustomPainter {
  _BevPainter(this.plan, this.hfovDeg);
  final PlanResult? plan;
  final double hfovDeg;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(Offset.zero & size, Paint()..color = const Color(0xFF151515));
    final p = plan;
    final spanX = kBevW * kCell, spanZ = kBevH * kCell;
    Offset at(double x, double z) => Offset((x - kX0) / spanX * size.width, size.height - z / spanZ * size.height);
    final cw = size.width / kBevW, ch = size.height / kBevH;

    if (p != null && p.ok && p.grid != null) {
      final g = p.grid!;
      // obstacle margin, same radius the planner uses (2 cells)
      final margin = List<bool>.filled(kBevW * kBevH, false);
      for (var cz = 0; cz < kBevH; cz++) {
        for (var cx = 0; cx < kBevW; cx++) {
          final v = g[cz * kBevW + cx];
          if (v != BevCell.obst && v != BevCell.drop) continue;
          for (var dz = -2; dz <= 2; dz++) {
            for (var dx = -2; dx <= 2; dx++) {
              final x = cx + dx, z = cz + dz;
              if (dx * dx + dz * dz <= 4 && x >= 0 && x < kBevW && z >= 0 && z < kBevH) margin[z * kBevW + x] = true;
            }
          }
        }
      }
      for (var cz = 0; cz < kBevH; cz++) {
        for (var cx = 0; cx < kBevW; cx++) {
          final v = g[cz * kBevW + cx];
          Color? c;
          if (v == BevCell.obst) {
            c = const Color(0xFFE53935);
          } else if (v == BevCell.drop) {
            c = const Color(0xFF8E24AA);
          } else if (margin[cz * kBevW + cx]) {
            c = const Color(0x66FFB300);
          } else if (v == BevCell.free) {
            c = const Color(0x882E7D32);
          }
          if (c == null) continue;
          canvas.drawRect(
            Rect.fromLTWH(cx * cw, size.height - (cz + 1) * ch, cw + 0.5, ch + 0.5),
            Paint()..color = c,
          );
        }
      }
    }

    // metre rings
    final ring = Paint()
      ..style = PaintingStyle.stroke
      ..color = Colors.white24;
    for (var m = 1; m <= 5; m++) {
      final r = m / spanZ * size.height;
      canvas.drawArc(Rect.fromCircle(center: at(0, 0), radius: r), -math.pi, math.pi, false, ring);
      final tp = TextPainter(
        text: TextSpan(text: '$m m', style: const TextStyle(color: Colors.white38, fontSize: 10)),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, at(0, m.toDouble()) + const Offset(3, -12));
    }
    // field of view
    final half = hfovDeg / 2 * math.pi / 180;
    final fov = Paint()
      ..color = Colors.white30
      ..strokeWidth = 1;
    canvas.drawLine(at(0, 0), at(spanZ * math.tan(half), spanZ), fov);
    canvas.drawLine(at(0, 0), at(-spanZ * math.tan(half), spanZ), fov);

    if (p != null && p.ok) {
      // sector hits
      final colours = [Colors.lightBlueAccent, Colors.white, Colors.orangeAccent];
      final hits = [p.left, p.center, p.right];
      for (var k = 0; k < 3; k++) {
        final h = hits[k];
        if (h == null) continue;
        final a = h.azimuthDeg * math.pi / 180;
        canvas.drawCircle(at(h.rangeM * math.sin(a), h.rangeM * math.cos(a)), 5, Paint()..color = colours[k]);
      }
      // path
      if (p.path.length > 1) {
        final line = Paint()
          ..color = Colors.cyanAccent
          ..strokeWidth = 3
          ..style = PaintingStyle.stroke;
        final path = Path()..moveTo(at(p.path.first.x, p.path.first.z).dx, at(p.path.first.x, p.path.first.z).dy);
        for (final w in p.path.skip(1)) {
          path.lineTo(at(w.x, w.z).dx, at(w.x, w.z).dy);
        }
        canvas.drawPath(path, line);
        for (final w in p.path) {
          canvas.drawCircle(at(w.x, w.z), 3, Paint()..color = Colors.cyanAccent);
        }
      }
      // goal
      if (p.goalX != null && p.goalZ != null) {
        final gp = at(p.goalX!.clamp(kX0, -kX0).toDouble(), p.goalZ!.clamp(0.0, spanZ).toDouble());
        final d = Path()
          ..moveTo(gp.dx, gp.dy - 7)
          ..lineTo(gp.dx + 7, gp.dy)
          ..lineTo(gp.dx, gp.dy + 7)
          ..lineTo(gp.dx - 7, gp.dy)
          ..close();
        canvas.drawPath(d, Paint()..color = p.hasGoal ? Colors.amber : Colors.amber.withOpacity(0.4));
      }
      // steering arrow
      final s = p.steerDeg;
      if (s != null) {
        final a = s * math.pi / 180;
        final tip = at(1.2 * math.sin(a), 1.2 * math.cos(a));
        final base = at(0, 0);
        final arrow = Paint()
          ..color = Colors.white
          ..strokeWidth = 4;
        canvas.drawLine(base, tip, arrow);
        canvas.drawCircle(tip, 5, Paint()..color = Colors.white);
      }
    }
    // the user
    final u = at(0, 0);
    canvas.drawPath(
      Path()
        ..moveTo(u.dx, u.dy - 12)
        ..lineTo(u.dx + 8, u.dy + 4)
        ..lineTo(u.dx - 8, u.dy + 4)
        ..close(),
      Paint()..color = Colors.white,
    );
  }

  @override
  bool shouldRepaint(covariant _BevPainter old) => true;
}
