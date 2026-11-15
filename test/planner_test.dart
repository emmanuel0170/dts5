import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:spatial_nav/planning/planner.dart';

// Each fixture is a synthetic scene rendered by tools/planner_prototype.py together with the answers the Python
// reference gave. The Dart port must reproduce them. Regenerate with: python3 tools/planner_prototype.py

Float32List _grid(List<dynamic> l) =>
    Float32List.fromList([for (final v in l) v == null ? double.nan : (v as num).toDouble()]);

void _near(double? actual, dynamic expected, double tol, String what) {
  if (expected == null) {
    expect(actual, isNull, reason: what);
  } else {
    expect(actual, isNotNull, reason: what);
    expect(actual!, closeTo((expected as num).toDouble(), tol), reason: what);
  }
}

void _sector(SectorHit? actual, dynamic expected, String what) {
  if (expected == null) {
    expect(actual, isNull, reason: what);
  } else {
    expect(actual, isNotNull, reason: what);
    expect(actual!.rangeM, closeTo((expected[0] as num).toDouble(), 0.15), reason: '$what range');
    expect(actual.azimuthDeg, closeTo((expected[1] as num).toDouble(), 3.0), reason: '$what azimuth');
  }
}

void main() {
  final dir = Directory('test/fixtures');
  final files = dir.existsSync() ? (dir.listSync().whereType<File>().where((f) => f.path.endsWith('.json')).toList()
    ..sort((a, b) => a.path.compareTo(b.path))) : <File>[];

  test('fixtures are present', () => expect(files, isNotEmpty));

  for (final f in files) {
    final name = f.uri.pathSegments.last.replaceAll('.json', '');
    test('matches the Python reference: $name', () {
      final j = jsonDecode(f.readAsStringSync()) as Map<String, dynamic>;
      final hfov = (j['hfov'] as num).toDouble();
      final planner = PathPlanner();
      final warm = j['warmup_depth'] as List?;
      if (warm != null) planner.plan(_grid(warm), hfov, nowMs: 0);
      final g = j['goal'] as List?;
      final goal = g == null ? null : PlanGoal((g[0] as num).toDouble(), (g[1] as num).toDouble(), (g[2] as num).toDouble());
      final r = planner.plan(_grid(j['depth'] as List), hfov, goal: goal, nowMs: warm != null ? 500 : 0);

      final e = j['expected'] as Map<String, dynamic>;
      expect(r.status.name, e['status']);
      if (r.status == PlanStatus.noGround) return;
      expect(r.heightM, closeTo((e['h'] as num).toDouble(), 0.02), reason: 'camera height');
      expect(r.pitchDeg, closeTo((e['pitch'] as num).toDouble(), 0.3), reason: 'pitch');
      _near(r.steerDeg, e['steer'], 1.0, 'steer');
      expect(r.reached, e['reached']);
      expect(r.held, e['held']);
      _near(r.obstacleAheadM, e['obstacle_ahead'], 0.01, 'obstacle ahead');
      _near(r.freeAheadM, e['free_ahead'], 0.01, 'free ahead');
      _near(r.openBearingDeg, e['open_bearing'], 0.01, 'open bearing');
      final s = e['sectors'] as Map<String, dynamic>;
      _sector(r.left, s['left'], 'left');
      _sector(r.center, s['center'], 'centre');
      _sector(r.right, s['right'], 'right');
      final path = e['path'] as List;
      expect(r.path.length, path.length, reason: 'waypoint count');
      expect(r.path.last.x, closeTo((path.last[0] as num).toDouble(), 0.01));
      expect(r.path.last.z, closeTo((path.last[1] as num).toDouble(), 0.01));
    });
  }
}
