import 'package:dts5/engine.dart';

FusedFrame frame(double tMs, {List<DetectionObj> dets = const [], List<double>? grid}) => FusedFrame(
      frameId: tMs.toInt(),
      tMs: tMs,
      w: 480,
      h: 640,
      detections: dets,
      depthFresh: true,
      depthAgeMs: 0,
      grid: grid,
      gridRows: grid == null ? 0 : 3,
      gridCols: grid == null ? 0 : 5,
      timing: const {},
    );

DetectionObj det(String label, {double cx = 0.5, double score = 0.9, double? depth, double? near}) =>
    DetectionObj(
      label: label,
      score: score,
      box: [cx - 0.1, 0.3, cx + 0.1, 0.7],
      depthMedian: depth,
      depthP10: near ?? depth,
    );

/// 3 rows x 5 columns, everything far except the given (row, col) cells.
List<double> gridWith(Map<(int, int), double> cells, {double fill = 5.0}) {
  final g = List<double>.filled(15, fill);
  cells.forEach((rc, v) => g[rc.$1 * 5 + rc.$2] = v);
  return g;
}
