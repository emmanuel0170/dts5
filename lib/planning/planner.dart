import 'dart:math' as math;
import 'dart:typed_data';

// Port of tools/planner_prototype.py, step for step (same xorshift RNG, same tie-breaking), so the fixtures that
// script writes give the same answers here. See that file for the reasoning behind every threshold.

const int kGridW = 48; // depth grid sent by the engine: 48 columns x 64 rows of the upright frame
const int kGridH = 64;
const double kCell = 0.2; // bird's-eye cell size, metres
const int kBevW = 28;
const int kBevH = 28;
const double kX0 = -kBevW * kCell / 2; // x of the left edge of the grid (the user stands at x = 0, z = 0)

class BevCell {
  static const int unknown = 0, free = 1, obst = 2, drop = 3;
}

const int _ransacIters = 80;
final double _minUpCos = math.cos(50 * math.pi / 180);
const double _groundRowFrac = 0.35;
const double _candZ0 = 0.4, _candZ1 = 6.0;
const double _minInlierFrac = 0.30;
const int _minInliers = 120;
const double _hMin = 0.4, _hMax = 2.6;
const double _overhead = 2.0;
const int _inflateCells = 2; // 0.4 m: half a body width plus margin
const double _nearUnknownM = 0.8;
const double _lookahead = 1.2;
const double _minGroundSpread = 1.0;
const double _planeAlpha = 0.4;
const double _holdMs = 2000;
const double _dropMaxZ = 5.0;
const double _minProgressM = 0.5;
const double _minPathM = 0.8;
const double _inf = 1e9;

double _tolGround(double z) => 0.07 + 0.04 * z;
double _obstacleMin(double z) => 0.18 + 0.04 * z;
double _dropMin(double z) => 0.25 + 0.05 * z;

/// xorshift32: identical integer arithmetic to the Python prototype.
class XorShift32 {
  XorShift32([int seed = 2463534242]) : _s = seed & 0xFFFFFFFF;
  int _s;

  int next() {
    var s = _s;
    s ^= (s << 13) & 0xFFFFFFFF;
    s ^= s >> 17;
    s ^= (s << 5) & 0xFFFFFFFF;
    _s = s & 0xFFFFFFFF;
    return _s;
  }

  int below(int n) => next() % n;
}

/// The target as a goal: its box centre in normalised image coordinates and its depth in metres.
class PlanGoal {
  const PlanGoal(this.xn, this.yn, this.depthM);
  final double xn, yn, depthM;
}

class SectorHit {
  const SectorHit(this.rangeM, this.azimuthDeg);
  final double rangeM;
  final double azimuthDeg; // 0 = straight ahead, negative = left
}

class PlanPoint {
  const PlanPoint(this.x, this.z);
  final double x; // metres to the right of the user
  final double z; // metres ahead
}

enum PlanStatus { ok, blocked, noGround }

class PlanResult {
  const PlanResult({
    required this.status,
    required this.tMs,
    this.held = false,
    this.heightM = 0,
    this.pitchDeg = 0,
    this.steerDeg,
    this.reached = false,
    this.path = const [],
    this.left,
    this.center,
    this.right,
    this.obstacleAheadM,
    this.freeAheadM,
    this.openBearingDeg,
    this.openRunM = 0,
    this.grid,
    this.goalX,
    this.goalZ,
    this.hasGoal = false,
  });

  final PlanStatus status;
  final double tMs; // frame time of the depth map this came from
  final bool held; // the ground plane is from a moment ago (too little floor in view now)
  final double heightM; // estimated camera height above the floor, in the depth model's own metres
  final double pitchDeg; // camera pitched down by this much
  final double? steerDeg; // direction to walk (0 = straight ahead), null if blocked
  final bool reached; // the path reaches the goal (false: it only gets as close as it can)
  final List<PlanPoint> path;
  final SectorHit? left, center, right; // nearest obstacle / drop in each lateral band
  final double? obstacleAheadM; // first obstacle or drop in a body-width corridor straight ahead
  final double? freeAheadM; // first non-free ground beyond 1 m
  final double? openBearingDeg; // bearing of the longest free run within +-30 deg (where to turn when blocked)
  final double openRunM;
  final Uint8List? grid; // kBevW x kBevH, row = z (0 nearest), col = x; values from BevCell
  final double? goalX, goalZ;
  final bool hasGoal; // the path was planned toward a target, not just straight ahead

  bool get ok => status != PlanStatus.noGround;

  static const noGround = PlanResult(status: PlanStatus.noGround, tMs: 0);
}

class _Fit {
  _Fit(this.nx, this.ny, this.nz, this.d, this.h, this.frac, this.abc);
  final double nx, ny, nz, d, h, frac;
  final List<double> abc;
}

class _Pts {
  final x = <double>[];
  final y = <double>[];
  final z = <double>[];
  final row = <int>[];
  int get n => x.length;
}

class _Item {
  _Item(this.c, this.x, this.z);
  final double c;
  final int x, z;
}

bool _less(_Item a, _Item b) => a.c != b.c ? a.c < b.c : (a.x != b.x ? a.x < b.x : a.z < b.z);

class _Heap {
  final _a = <_Item>[];
  bool get isEmpty => _a.isEmpty;

  void push(_Item it) {
    _a.add(it);
    var i = _a.length - 1;
    while (i > 0) {
      final p = (i - 1) >> 1;
      if (_less(_a[i], _a[p])) {
        final t = _a[i];
        _a[i] = _a[p];
        _a[p] = t;
        i = p;
      } else {
        break;
      }
    }
  }

  _Item pop() {
    final top = _a.first;
    final last = _a.removeLast();
    if (_a.isNotEmpty) {
      _a[0] = last;
      var i = 0;
      while (true) {
        final l = 2 * i + 1, r = l + 1;
        var m = i;
        if (l < _a.length && _less(_a[l], _a[m])) m = l;
        if (r < _a.length && _less(_a[r], _a[m])) m = r;
        if (m == i) break;
        final t = _a[i];
        _a[i] = _a[m];
        _a[m] = t;
        i = m;
      }
    }
    return top;
  }
}

/// Local path planning from one metric depth map, with no pose and no map: ground plane -> obstacle / drop
/// classification -> bird's-eye occupancy -> path toward a goal. Stateful only for the ground-plane smoothing and
/// the short hold when too little floor is visible.
class PathPlanner {
  final XorShift32 _rng = XorShift32();
  List<double>? _abc;
  double _abcT = 0;

  /// [depth]: kGridW x kGridH metres, row-major, row 0 = top of the upright frame, NaN = invalid.
  /// [hfovDeg]: horizontal field of view of that frame.
  PlanResult plan(Float32List depth, double hfovDeg, {PlanGoal? goal, double nowMs = 0}) {
    final hfov = (hfovDeg.isFinite && hfovDeg > 20 && hfovDeg < 140) ? hfovDeg : 60.0;
    final pts = _unproject(depth, hfov);
    final fit = _fitGround(pts);
    var held = false;
    if (fit != null) {
      var blended = fit.abc;
      final old = _abc;
      if (old != null) {
        blended = [for (var i = 0; i < 3; i++) old[i] + _planeAlpha * (blended[i] - old[i])];
      }
      _abc = blended;
      _abcT = nowMs;
    } else if (_abc != null && nowMs - _abcT <= _holdMs) {
      held = true; // not enough floor in view right now: keep the last good plane for a moment
    } else {
      _abc = null;
      return PlanResult(status: PlanStatus.noGround, tMs: nowMs);
    }
    final abc = _abc!;
    final lm = math.sqrt(abc[0] * abc[0] + 1.0 + abc[1] * abc[1]);
    final nx = abc[0] / lm, ny = -1.0 / lm, nz = abc[1] / lm;
    final d = -abc[2] / lm;
    final h = -d;

    final bev = _buildBev(pts, nx, ny, nz, d);
    final grid = bev.grid;

    double gx, gz;
    if (goal != null) {
      final f = (kGridW / 2) / math.tan(hfov * math.pi / 360);
      final x = (goal.xn * kGridW - kGridW / 2) * goal.depthM / f;
      final y = (goal.yn * kGridH - kGridH / 2) * goal.depthM / f;
      gx = bev.r[0] * x + bev.r[1] * y + bev.r[2] * goal.depthM;
      gz = bev.f[0] * x + bev.f[1] * y + bev.f[2] * goal.depthM;
    } else {
      gx = 0.0;
      gz = 4.0;
    }

    final pr = _planOnGrid(grid, gx, gz);
    final path = pr.path;
    var length = 0.0;
    for (var i = 0; i + 1 < path.length; i++) {
      length += math.sqrt(math.pow(path[i + 1].x - path[i].x, 2) + math.pow(path[i + 1].z - path[i].z, 2));
    }
    final progress = math.sqrt(gx * gx + gz * gz) -
        math.sqrt(math.pow(path.last.x - gx, 2) + math.pow(path.last.z - gz, 2));
    final blocked = !pr.reached && (length < _minPathM || progress < _minProgressM);
    final steer = blocked ? null : _steerBearing(path);
    final goalBearing = math.atan2(gx, gz) * 180 / math.pi;
    final ob = _openBearing(grid, goalBearing);
    final ah = _ahead(grid);
    final sec = _sectors(grid);

    return PlanResult(
      status: blocked ? PlanStatus.blocked : PlanStatus.ok,
      tMs: nowMs,
      held: held,
      heightM: h,
      pitchDeg: math.atan2(-nz, -ny) * 180 / math.pi,
      steerDeg: steer,
      reached: pr.reached,
      path: path,
      left: sec[0],
      center: sec[1],
      right: sec[2],
      obstacleAheadM: ah[0],
      freeAheadM: ah[1],
      openBearingDeg: ob.$1,
      openRunM: ob.$2,
      grid: grid,
      goalX: gx,
      goalZ: gz,
      hasGoal: goal != null,
    );
  }

  // ------------------------------------------------------------------ geometry

  _Pts _unproject(Float32List depth, double hfov) {
    final f = (kGridW / 2) / math.tan(hfov * math.pi / 360);
    final p = _Pts();
    for (var j = 0; j < kGridH; j++) {
      for (var i = 0; i < kGridW; i++) {
        final z = depth[j * kGridW + i].toDouble();
        if (z.isNaN || z < 0.3 || z > 12.0) continue;
        p.x.add((i + 0.5 - kGridW / 2) * z / f);
        p.y.add((j + 0.5 - kGridH / 2) * z / f);
        p.z.add(z);
        p.row.add(j);
      }
    }
    return p;
  }

  List<double>? _solve3(List<List<double>> a, List<double> b) {
    final m = [
      [a[0][0], a[0][1], a[0][2], b[0]],
      [a[1][0], a[1][1], a[1][2], b[1]],
      [a[2][0], a[2][1], a[2][2], b[2]],
    ];
    for (var c = 0; c < 3; c++) {
      var p = c;
      for (var r = c + 1; r < 3; r++) {
        if (m[r][c].abs() > m[p][c].abs()) p = r; // first maximum wins, like Python's max()
      }
      if (m[p][c].abs() < 1e-12) return null;
      final t = m[c];
      m[c] = m[p];
      m[p] = t;
      for (var r = c + 1; r < 3; r++) {
        final f = m[r][c] / m[c][c];
        for (var k = c; k < 4; k++) {
          m[r][k] -= f * m[c][k];
        }
      }
    }
    final x = [0.0, 0.0, 0.0];
    for (var r = 2; r >= 0; r--) {
      var s = m[r][3];
      for (var k = r + 1; k < 3; k++) {
        s -= m[r][k] * x[k];
      }
      x[r] = s / m[r][r];
    }
    return x;
  }

  _Fit? _fitGround(_Pts p) {
    final rowMin = (_groundRowFrac * kGridH).ceil();
    final cand = <int>[];
    for (var k = 0; k < p.n; k++) {
      if (p.row[k] >= rowMin && p.z[k] >= _candZ0 && p.z[k] <= _candZ1) cand.add(k);
    }
    if (cand.length < _minInliers) return null;

    int countInliers(double nx, double ny, double nz, double d) {
      var cnt = 0;
      for (final k in cand) {
        if ((nx * p.x[k] + ny * p.y[k] + nz * p.z[k] - d).abs() < _tolGround(p.z[k])) cnt++;
      }
      return cnt;
    }

    var bestN = 0;
    double bnx = 0, bny = 0, bnz = 0, bd = 0;
    var have = false;
    for (var it = 0; it < _ransacIters; it++) {
      final a = _rng.below(cand.length), b = _rng.below(cand.length), c = _rng.below(cand.length);
      if (a == b || b == c || a == c) continue;
      final k1 = cand[a], k2 = cand[b], k3 = cand[c];
      final ux = p.x[k2] - p.x[k1], uy = p.y[k2] - p.y[k1], uz = p.z[k2] - p.z[k1];
      final vx = p.x[k3] - p.x[k1], vy = p.y[k3] - p.y[k1], vz = p.z[k3] - p.z[k1];
      var nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      final ln = math.sqrt(nx * nx + ny * ny + nz * nz);
      if (ln < 1e-9) continue;
      nx /= ln;
      ny /= ln;
      nz /= ln;
      if (ny > 0) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
      }
      if (-ny < _minUpCos) continue;
      final d = nx * p.x[k1] + ny * p.y[k1] + nz * p.z[k1];
      final cnt = countInliers(nx, ny, nz, d);
      if (cnt > bestN) {
        bestN = cnt;
        bnx = nx;
        bny = ny;
        bnz = nz;
        bd = d;
        have = true;
      }
    }
    if (!have) return null;

    var nx = bnx, ny = bny, nz = bnz, d = bd;
    for (var r = 0; r < 2; r++) {
      // refine: least squares y = a x + b z + c over the inliers, then re-collect them
      double sxx = 0, sxz = 0, sx = 0, szz = 0, sz = 0, sxy = 0, szy = 0, sy = 0;
      var n = 0;
      for (final k in cand) {
        if ((nx * p.x[k] + ny * p.y[k] + nz * p.z[k] - d).abs() < _tolGround(p.z[k])) {
          final x = p.x[k], y = p.y[k], z = p.z[k];
          sxx += x * x;
          sxz += x * z;
          sx += x;
          szz += z * z;
          sz += z;
          sxy += x * y;
          szy += z * y;
          sy += y;
          n++;
        }
      }
      if (n < 3) return null;
      final sol = _solve3([
        [sxx, sxz, sx],
        [sxz, szz, sz],
        [sx, sz, n.toDouble()],
      ], [sxy, szy, sy]);
      if (sol == null) return null;
      final a = sol[0], b = sol[1], c = sol[2];
      final lm = math.sqrt(a * a + 1.0 + b * b);
      nx = a / lm;
      ny = -1.0 / lm;
      nz = b / lm;
      d = -c / lm;
    }

    var inliers = 0;
    var zMin = double.infinity, zMax = -double.infinity;
    for (final k in cand) {
      if ((nx * p.x[k] + ny * p.y[k] + nz * p.z[k] - d).abs() < _tolGround(p.z[k])) {
        inliers++;
        if (p.z[k] < zMin) zMin = p.z[k];
        if (p.z[k] > zMax) zMax = p.z[k];
      }
    }
    final frac = inliers / cand.length;
    final h = -d;
    if (frac < _minInlierFrac || inliers < _minInliers || h < _hMin || h > _hMax) return null;
    if (zMax - zMin < _minGroundSpread) return null;
    return _Fit(nx, ny, nz, d, h, frac, [-nx / ny, -nz / ny, (-d) / (-ny)]);
  }

  // ------------------------------------------------------------------ bird's-eye grid

  static (int, int) _cellOf(double x, double z) => (((x - kX0) / kCell).floor(), (z / kCell).floor());
  static bool _inside(int cx, int cz) => cx >= 0 && cx < kBevW && cz >= 0 && cz < kBevH;

  /// Cells on the line from (x0, y0) to (x1, y1), inclusive, as a flat [x, y, x, y, ...] list.
  static List<int> _line(int x0, int y0, int x1, int y1) {
    final out = <int>[];
    final dx = (x1 - x0).abs();
    final dy = -(y1 - y0).abs();
    final sx = x0 < x1 ? 1 : -1;
    final sy = y0 < y1 ? 1 : -1;
    var err = dx + dy;
    while (true) {
      out.add(x0);
      out.add(y0);
      if (x0 == x1 && y0 == y1) break;
      final e2 = 2 * err;
      if (e2 >= dy) {
        err += dy;
        x0 += sx;
      }
      if (e2 <= dx) {
        err += dx;
        y0 += sy;
      }
    }
    return out;
  }

  ({Uint8List grid, List<double> f, List<double> r}) _buildBev(_Pts p, double nx, double ny, double nz, double d) {
    var fx = -nz * nx, fy = -nz * ny, fz = 1 - nz * nz;
    final lf = math.sqrt(fx * fx + fy * fy + fz * fz);
    fx /= lf;
    fy /= lf;
    fz /= lf;
    final rx = fy * nz - fz * ny, ry = fz * nx - fx * nz, rz = fx * ny - fy * nx;

    final occ = List<int>.filled(kBevW * kBevH, 0);
    final drp = List<int>.filled(kBevW * kBevH, 0);
    final fre = List<int>.filled(kBevW * kBevH, 0);
    final (sx0, sz0) = _cellOf(0.0, 0.0);

    for (var k = 0; k < p.n; k++) {
      final x = p.x[k], y = p.y[k], z = p.z[k];
      final hgt = nx * x + ny * y + nz * z - d;
      final bx = rx * x + ry * y + rz * z;
      final bz = fx * x + fy * y + fz * z;
      final (cx, cz) = _cellOf(bx, bz);
      if (!_inside(cx, cz) || bz < 0) continue;
      final tol = _tolGround(z);
      final String kind;
      if (hgt.abs() <= tol) {
        kind = 'g';
      } else if (tol < hgt && hgt > _obstacleMin(z) && hgt <= _overhead) {
        kind = 'o';
      } else if (hgt < -_dropMin(z) && z <= _dropMaxZ) {
        kind = 'd';
      } else {
        continue;
      }
      if (kind == 'd') {
        // The ground should have been seen where this ray crosses the plane, but the surface is lower (a hole,
        // a step down): everything from that crossing to here is a drop. Such a ray passes OVER the hole, so it
        // must not carve free space.
        final npd = nx * x + ny * y + nz * z;
        var s0 = npd < -1e-9 ? d / npd : 1.0;
        s0 = math.min(math.max(s0, 0.0), 1.0);
        final (ex, ez) = _cellOf(rx * x * s0 + ry * y * s0 + rz * z * s0, fx * x * s0 + fy * y * s0 + fz * z * s0);
        final line = _line(ex, ez, cx, cz);
        for (var i = 0; i < line.length; i += 2) {
          if (_inside(line[i], line[i + 1])) drp[line[i + 1] * kBevW + line[i]]++;
        }
        continue;
      }
      final line = _line(sx0, sz0, cx, cz);
      for (var i = 0; i < line.length; i += 2) {
        final qx = line[i], qz = line[i + 1];
        if (qx == cx && qz == cz && kind != 'g') continue;
        if (_inside(qx, qz)) fre[qz * kBevW + qx]++;
      }
      if (kind == 'o') occ[cz * kBevW + cx]++;
    }

    final grid = Uint8List(kBevW * kBevH);
    for (var k = 0; k < grid.length; k++) {
      if (occ[k] >= 2 || (occ[k] >= 1 && fre[k] < 2)) {
        grid[k] = BevCell.obst;
      } else if (drp[k] >= 1) {
        grid[k] = BevCell.drop; // explicit evidence the surface is lower: wins over carved free space
      } else if (fre[k] >= 1) {
        grid[k] = BevCell.free;
      }
    }
    // an isolated drop cell is noise: real drops come as runs along the ray
    for (var cz = 0; cz < kBevH; cz++) {
      for (var cx = 0; cx < kBevW; cx++) {
        if (grid[cz * kBevW + cx] != BevCell.drop) continue;
        var neighbour = false;
        for (var dz = -1; dz <= 1 && !neighbour; dz++) {
          for (var dx = -1; dx <= 1; dx++) {
            if ((dx != 0 || dz != 0) &&
                _inside(cx + dx, cz + dz) &&
                grid[(cz + dz) * kBevW + cx + dx] == BevCell.drop) {
              neighbour = true;
              break;
            }
          }
        }
        if (!neighbour) {
          grid[cz * kBevW + cx] = fre[cz * kBevW + cx] >= 1 ? BevCell.free : BevCell.unknown;
        }
      }
    }
    return (grid: grid, f: [fx, fy, fz], r: [rx, ry, rz]);
  }

  // ------------------------------------------------------------------ planning

  static const _nb = <(int, int, double)>[
    (-1, -1, 1.4142), (0, -1, 1.0), (1, -1, 1.4142), (-1, 0, 1.0), (1, 0, 1.0), (-1, 1, 1.4142), (0, 1, 1.0), (1, 1, 1.4142),
  ];

  ({List<PlanPoint> path, bool reached}) _planOnGrid(Uint8List grid, double gx, double gz) {
    final (sx, sz) = _cellOf(0.0, 0.0);
    final n = kBevW * kBevH;
    final blockedRaw = List<bool>.generate(n, (k) => grid[k] == BevCell.obst || grid[k] == BevCell.drop);
    final inflated = List<bool>.filled(n, false);
    for (var cz = 0; cz < kBevH; cz++) {
      for (var cx = 0; cx < kBevW; cx++) {
        if (!blockedRaw[cz * kBevW + cx]) continue;
        for (var dz = -_inflateCells; dz <= _inflateCells; dz++) {
          for (var dx = -_inflateCells; dx <= _inflateCells; dx++) {
            if (dx * dx + dz * dz <= _inflateCells * _inflateCells && _inside(cx + dx, cz + dz)) {
              inflated[(cz + dz) * kBevW + cx + dx] = true;
            }
          }
        }
      }
    }
    // distance (cells) to the nearest raw obstacle, for a soft clearance cost
    final dist = List<double>.generate(n, (k) => blockedRaw[k] ? 0.0 : _inf);
    final q = <int>[for (var k = 0; k < n; k++) if (blockedRaw[k]) k];
    for (var qi = 0; qi < q.length; qi++) {
      final k = q[qi];
      final cx = k % kBevW, cz = k ~/ kBevW;
      for (final (dx, dz, w) in _nb) {
        final nx = cx + dx, nz = cz + dz;
        if (_inside(nx, nz) && dist[nz * kBevW + nx] > dist[k] + w) {
          dist[nz * kBevW + nx] = dist[k] + w;
          q.add(nz * kBevW + nx);
        }
      }
    }

    bool traversable(int cx, int cz) {
      final k = cz * kBevW + cx;
      if (cx == sx && cz == sz) return true;
      if (inflated[k]) return false;
      if (grid[k] == BevCell.free) return true;
      if (grid[k] == BevCell.unknown) {
        final x = kX0 + (cx + 0.5) * kCell;
        final z = (cz + 0.5) * kCell;
        return math.sqrt(x * x + z * z) <= _nearUnknownM;
      }
      return false;
    }

    final cost = List<double>.filled(n, _inf);
    final parent = List<int>.filled(n, -1);
    final order = <int>[]; // cells in the order they first received a cost (Python dict insertion order)
    final startK = sz * kBevW + sx;
    cost[startK] = 0.0;
    order.add(startK);
    final heap = _Heap()..push(_Item(0.0, sx, sz));
    while (!heap.isEmpty) {
      final it = heap.pop();
      final cx = it.x, cz = it.z;
      if (it.c > cost[cz * kBevW + cx]) continue;
      for (final (dx, dz, w) in _nb) {
        final nx = cx + dx, nz = cz + dz;
        if (!_inside(nx, nz) || !traversable(nx, nz)) continue;
        if (dx != 0 && dz != 0 && !(traversable(cx + dx, cz) && traversable(cx, cz + dz))) continue;
        final pen = math.max(0.0, 3.0 - dist[nz * kBevW + nx]) * 0.5;
        final nc = it.c + w + pen;
        final nk = nz * kBevW + nx;
        if (nc < cost[nk]) {
          if (cost[nk] >= _inf) order.add(nk);
          cost[nk] = nc;
          parent[nk] = cz * kBevW + cx;
          heap.push(_Item(nc, nx, nz));
        }
      }
    }

    var (gcx, gcz) = _cellOf(gx, gz);
    gcx = math.min(math.max(gcx, 0), kBevW - 1);
    gcz = math.min(math.max(gcz, 0), kBevH - 1);
    final goalK = gcz * kBevW + gcx;
    int endK;
    if (cost[goalK] < _inf) {
      endK = goalK;
    } else {
      endK = order.first;
      var bestD = double.infinity, bestC = double.infinity;
      for (final k in order) {
        final dx = (k % kBevW - gcx).toDouble(), dz = (k ~/ kBevW - gcz).toDouble();
        final dd = math.sqrt(dx * dx + dz * dz);
        if (dd < bestD || (dd == bestD && cost[k] < bestC)) {
          bestD = dd;
          bestC = cost[k];
          endK = k;
        }
      }
    }
    final cells = <int>[endK];
    while (cells.last != startK) {
      cells.add(parent[cells.last]);
    }
    final path = cells.reversed.toList();

    bool clearLine(int a, int b) {
      final line = _line(a % kBevW, a ~/ kBevW, b % kBevW, b ~/ kBevW);
      for (var i = 0; i < line.length; i += 2) {
        if (!traversable(line[i], line[i + 1])) return false;
      }
      return true;
    }

    final way = <int>[path.first];
    var i = 0;
    while (i < path.length - 1) {
      var j = path.length - 1;
      while (j > i + 1 && !clearLine(path[i], path[j])) {
        j--;
      }
      way.add(path[j]);
      i = j;
    }
    final pts = [
      for (final k in way) PlanPoint(kX0 + (k % kBevW + 0.5) * kCell, (k ~/ kBevW + 0.5) * kCell),
    ];
    pts[0] = const PlanPoint(0.0, 0.0);
    return (path: pts, reached: endK == goalK);
  }

  double? _steerBearing(List<PlanPoint> path) {
    var need = _lookahead;
    for (var k = 0; k + 1 < path.length; k++) {
      final a = path[k], b = path[k + 1];
      final seg = math.sqrt(math.pow(b.x - a.x, 2) + math.pow(b.z - a.z, 2));
      if (seg >= need && seg > 1e-9) {
        final t = need / seg;
        final x = a.x + (b.x - a.x) * t, z = a.z + (b.z - a.z) * t;
        return math.atan2(x, z) * 180 / math.pi;
      }
      need -= seg;
    }
    final last = path.last;
    if (math.sqrt(last.x * last.x + last.z * last.z) < 0.3) return null;
    return math.atan2(last.x, last.z) * 180 / math.pi;
  }

  // ------------------------------------------------------------------ outputs for cues

  /// Nearest raw obstacle / drop in the left, centre (a body-width corridor) and right lateral bands.
  List<SectorHit?> _sectors(Uint8List grid) {
    final out = <SectorHit?>[null, null, null];
    for (var cz = 0; cz < kBevH; cz++) {
      for (var cx = 0; cx < kBevW; cx++) {
        final g = grid[cz * kBevW + cx];
        if (g != BevCell.obst && g != BevCell.drop) continue;
        final x = kX0 + (cx + 0.5) * kCell;
        final z = (cz + 0.5) * kCell;
        final r = math.sqrt(x * x + z * z);
        if (r > 3.0 || x.abs() > 2.0) continue;
        final key = x.abs() <= 0.5 ? 1 : (x < 0 ? 0 : 2);
        final cur = out[key];
        if (cur == null || r < cur.rangeM) out[key] = SectorHit(r, math.atan2(x, z) * 180 / math.pi);
      }
    }
    return out;
  }

  List<double?> _ahead(Uint8List grid) {
    double? obst, free;
    for (var cz = 0; cz < kBevH; cz++) {
      var rowBlocked = false, rowFree = false;
      for (var cx = 0; cx < kBevW; cx++) {
        final x = kX0 + (cx + 0.5) * kCell;
        if (x.abs() > 0.35) continue;
        final g = grid[cz * kBevW + cx];
        if (g == BevCell.obst || g == BevCell.drop) rowBlocked = true;
        if (g == BevCell.free) rowFree = true;
      }
      final z = cz * kCell;
      if (rowBlocked && obst == null) obst = z;
      if (!rowFree && !rowBlocked && free == null && z >= 1.0) free = z;
    }
    return [obst, free];
  }

  (double?, double) _openBearing(Uint8List grid, double goalBearing) {
    double? bestDeg;
    var bestScore = -1.0;
    for (var deg = -30; deg <= 30; deg += 5) {
      final a = deg * math.pi / 180;
      var run = 0.0;
      var r = 0.2;
      while (r < kBevH * kCell) {
        final (cx, cz) = _cellOf(r * math.sin(a), r * math.cos(a));
        if (!_inside(cx, cz) || grid[cz * kBevW + cx] != BevCell.free) {
          if (!(r <= _nearUnknownM && _inside(cx, cz) && grid[cz * kBevW + cx] == BevCell.unknown)) break;
        }
        run = r;
        r += 0.2;
      }
      final score = run - 0.002 * (deg - goalBearing).abs();
      if (score > bestScore) {
        bestScore = score;
        bestDeg = deg.toDouble();
      }
    }
    return (bestDeg, math.max(bestScore, 0.0));
  }
}
