"""Reference implementation of the local path planner, validated on synthetic scenes.

The Dart port (lib/planning/planner.dart) follows this file step for step, including the xorshift RNG, so the
fixtures this script writes (test/fixtures/*.json) must give the same answers in Dart.

Pipeline (per fresh depth map, no pose / no SLAM):
  1. unproject the 48x64 metric depth grid with a pinhole model (hfov from the camera)
  2. fit the ground plane with RANSAC + least squares (no IMU needed); gate it on plausibility
  3. classify every point: ground / obstacle / drop (height above the plane)
  4. build a bird's-eye occupancy grid (0.2 m cells, 5.6 x 5.6 m) by splatting + ray carving
  5. inflate obstacles by the user's half width, plan with Dijkstra toward a goal (target or straight ahead),
     pull the string, take a 1.2 m lookahead as the steering direction
  6. report the nearest obstacle in left / centre / right lateral bands (used for directional sonification)

Run:  python3 tools/planner_prototype.py            (runs the checks, writes test/fixtures)
"""
import heapq
import json
import math
import pathlib

W, H = 48, 64
CELL = 0.2
BEV_W, BEV_H = 28, 28
X0 = -BEV_W * CELL / 2  # x of the left edge of the grid
UNKNOWN, FREE, OBST, DROP = 0, 1, 2, 3

# ------------------------------------------------------------------ parameters
RANSAC_ITERS = 80
MIN_UP_COS = math.cos(math.radians(50))
GROUND_ROW_FRAC = 0.35
CAND_Z = (0.4, 6.0)
MIN_INLIER_FRAC = 0.30
MIN_INLIERS = 120
H_RANGE = (0.4, 2.6)
OVERHEAD = 2.0
INFLATE_CELLS = 2  # 0.4 m: half a body width plus margin
NEAR_UNKNOWN_M = 0.8
LOOKAHEAD = 1.2
MIN_GROUND_SPREAD = 1.0  # inliers must span this much depth: a sliver of floor in front of a wall is not a plane
PLANE_ALPHA = 0.4
HOLD_MS = 2000
DROP_MAX_Z = 5.0
MIN_PROGRESS_M = 0.5
MIN_PATH_M = 0.8


class Rng:
    """xorshift32: identical integer arithmetic in Python and Dart."""

    def __init__(self, seed=2463534242):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        s = self.s
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        self.s = s & 0xFFFFFFFF
        return self.s

    def below(self, n):
        return self.next() % n


def tol_ground(z):
    return 0.07 + 0.04 * z


def obstacle_min(z):
    return 0.18 + 0.04 * z


def drop_min(z):
    return 0.25 + 0.05 * z


def solve3(a, b):
    """Gaussian elimination with partial pivoting for a 3x3 system."""
    m = [a[0][:] + [b[0]], a[1][:] + [b[1]], a[2][:] + [b[2]]]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            return None
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, 3):
            f = m[r][c] / m[c][c]
            for k in range(c, 4):
                m[r][k] -= f * m[c][k]
    x = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        x[r] = (m[r][3] - sum(m[r][k] * x[k] for k in range(r + 1, 3))) / m[r][r]
    return x


def unproject(depth, hfov_deg):
    f = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
    pts = []  # (x, y, z, row)
    for j in range(H):
        for i in range(W):
            z = depth[j * W + i]
            if not (z == z) or z < 0.3 or z > 12.0:
                continue
            pts.append(((i + 0.5 - W / 2) * z / f, (j + 0.5 - H / 2) * z / f, z, j))
    return pts


def fit_ground(pts, rng):
    cand = [p for p in pts if p[3] >= int(math.ceil(GROUND_ROW_FRAC * H)) and CAND_Z[0] <= p[2] <= CAND_Z[1]]
    if len(cand) < MIN_INLIERS:
        return None
    best, best_n = None, 0
    for _ in range(RANSAC_ITERS):
        a, b, c = rng.below(len(cand)), rng.below(len(cand)), rng.below(len(cand))
        if a == b or b == c or a == c:
            continue
        p1, p2, p3 = cand[a], cand[b], cand[c]
        u = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        v = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
        n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        ln = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        if ln < 1e-9:
            continue
        n = (n[0] / ln, n[1] / ln, n[2] / ln)
        if n[1] > 0:
            n = (-n[0], -n[1], -n[2])
        if -n[1] < MIN_UP_COS:
            continue
        d = n[0] * p1[0] + n[1] * p1[1] + n[2] * p1[2]
        cnt = 0
        for q in cand:
            if abs(n[0] * q[0] + n[1] * q[1] + n[2] * q[2] - d) < tol_ground(q[2]):
                cnt += 1
        if cnt > best_n:
            best_n, best = cnt, (n, d)
    if best is None:
        return None
    plane = best
    for _ in range(2):  # refine: least squares y = a x + b z + c over the inliers, then re-collect inliers
        n, d = plane
        inl = [q for q in cand if abs(n[0] * q[0] + n[1] * q[1] + n[2] * q[2] - d) < tol_ground(q[2])]
        if len(inl) < 3:
            return None
        sxx = sxz = sx = szz = sz = sxy = szy = sy = 0.0
        for q in inl:
            sxx += q[0] * q[0]; sxz += q[0] * q[2]; sx += q[0]; szz += q[2] * q[2]; sz += q[2]
            sxy += q[0] * q[1]; szy += q[2] * q[1]; sy += q[1]
        sol = solve3([[sxx, sxz, sx], [sxz, szz, sz], [sx, sz, float(len(inl))]], [sxy, szy, sy])
        if sol is None:
            return None
        a, b, c = sol
        m = (a, -1.0, b)
        lm = math.sqrt(a * a + 1.0 + b * b)
        plane = ((m[0] / lm, m[1] / lm, m[2] / lm), -c / lm)
    n, d = plane
    inl = [q for q in cand if abs(n[0] * q[0] + n[1] * q[1] + n[2] * q[2] - d) < tol_ground(q[2])]
    inliers = len(inl)
    frac = inliers / len(cand)
    h = -d
    if frac < MIN_INLIER_FRAC or inliers < MIN_INLIERS or not (H_RANGE[0] <= h <= H_RANGE[1]):
        return None
    if max(q[2] for q in inl) - min(q[2] for q in inl) < MIN_GROUND_SPREAD:
        return None
    return n, d, h, frac, (-n[0] / n[1], -n[2] / n[1], -d / -n[1])


def bresenham(x0, y0, x1, y1):
    cells = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return cells


def cell_of(x, z):
    cx = int(math.floor((x - X0) / CELL))
    cz = int(math.floor(z / CELL))
    return cx, cz


def inside(cx, cz):
    return 0 <= cx < BEV_W and 0 <= cz < BEV_H


def build_bev(pts, n, d):
    nx, ny, nz = n
    fx, fy, fz = -nz * nx, -nz * ny, 1 - nz * nz
    lf = math.sqrt(fx * fx + fy * fy + fz * fz)
    fx, fy, fz = fx / lf, fy / lf, fz / lf
    rx, ry, rz = fy * nz - fz * ny, fz * nx - fx * nz, fx * ny - fy * nx
    occ = [0] * (BEV_W * BEV_H)
    drp = [0] * (BEV_W * BEV_H)
    fre = [0] * (BEV_W * BEV_H)
    sx0, sz0 = cell_of(0.0, 0.0)
    for (x, y, z, _row) in pts:
        hgt = nx * x + ny * y + nz * z - d
        bx = rx * x + ry * y + rz * z
        bz = fx * x + fy * y + fz * z
        cx, cz = cell_of(bx, bz)
        if not inside(cx, cz) or bz < 0:
            continue
        tol = tol_ground(z)
        if abs(hgt) <= tol:
            kind = 'g'
        elif tol < hgt and hgt > obstacle_min(z) and hgt <= OVERHEAD:
            kind = 'o'
        elif hgt < -drop_min(z) and z <= DROP_MAX_Z:
            kind = 'd'
        else:
            continue
        if kind == 'd':
            # The ground should have been seen where this ray crosses the plane, but the surface is lower
            # (a hole, a step down, a kerb): everything from that crossing to here is a drop. Such a ray must not
            # carve free space: it passes OVER the hole.
            npd = nx * x + ny * y + nz * z
            s0 = d / npd if npd < -1e-9 else 1.0
            s0 = min(max(s0, 0.0), 1.0)
            ex, ez = cell_of(rx * x * s0 + ry * y * s0 + rz * z * s0, fx * x * s0 + fy * y * s0 + fz * z * s0)
            for (qx, qz) in bresenham(ex, ez, cx, cz):
                if inside(qx, qz):
                    drp[qz * BEV_W + qx] += 1
            continue
        for (qx, qz) in bresenham(sx0, sz0, cx, cz):
            if (qx, qz) == (cx, cz) and kind != 'g':
                continue
            if inside(qx, qz):
                fre[qz * BEV_W + qx] += 1
        if kind == 'o':
            occ[cz * BEV_W + cx] += 1
    grid = [UNKNOWN] * (BEV_W * BEV_H)
    for k in range(BEV_W * BEV_H):
        if occ[k] >= 2 or (occ[k] >= 1 and fre[k] < 2):
            grid[k] = OBST
        elif drp[k] >= 1:
            grid[k] = DROP  # explicit evidence the surface is lower: it wins over free space carved by rays passing over it
        elif fre[k] >= 1:
            grid[k] = FREE
    # an isolated drop cell is noise: real drops come as runs along the ray
    for cz in range(BEV_H):
        for cx in range(BEV_W):
            if grid[cz * BEV_W + cx] != DROP:
                continue
            if not any(inside(cx + dx, cz + dz) and (dx or dz) and grid[(cz + dz) * BEV_W + cx + dx] == DROP
                       for dz in (-1, 0, 1) for dx in (-1, 0, 1)):
                grid[cz * BEV_W + cx] = FREE if fre[cz * BEV_W + cx] >= 1 else UNKNOWN
    return grid, (fx, fy, fz), (rx, ry, rz)


def neighbors8():
    return [(-1, -1, 1.4142), (0, -1, 1.0), (1, -1, 1.4142), (-1, 0, 1.0), (1, 0, 1.0), (-1, 1, 1.4142), (0, 1, 1.0), (1, 1, 1.4142)]


def plan_on_grid(grid, goal_xz):
    sx, sz = cell_of(0.0, 0.0)
    blocked_raw = [g in (OBST, DROP) for g in grid]
    # inflation
    inflated = [False] * len(grid)
    for cz in range(BEV_H):
        for cx in range(BEV_W):
            if not blocked_raw[cz * BEV_W + cx]:
                continue
            for dz in range(-INFLATE_CELLS, INFLATE_CELLS + 1):
                for dx in range(-INFLATE_CELLS, INFLATE_CELLS + 1):
                    if dx * dx + dz * dz <= INFLATE_CELLS * INFLATE_CELLS and inside(cx + dx, cz + dz):
                        inflated[(cz + dz) * BEV_W + cx + dx] = True
    # distance to nearest raw obstacle (chamfer, in cells), for a soft clearance cost
    INF = 1e9
    dist = [0.0 if b else INF for b in blocked_raw]
    q = [i for i, b in enumerate(blocked_raw) if b]
    qi = 0
    while qi < len(q):
        k = q[qi]; qi += 1
        cx, cz = k % BEV_W, k // BEV_W
        for dx, dz, w in neighbors8():
            nx_, nz_ = cx + dx, cz + dz
            if inside(nx_, nz_) and dist[nz_ * BEV_W + nx_] > dist[k] + w:
                dist[nz_ * BEV_W + nx_] = dist[k] + w
                q.append(nz_ * BEV_W + nx_)

    def traversable(cx, cz):
        k = cz * BEV_W + cx
        if (cx, cz) == (sx, sz):
            return True
        if inflated[k]:
            return False
        if grid[k] == FREE:
            return True
        if grid[k] == UNKNOWN:
            x = X0 + (cx + 0.5) * CELL
            z = (cz + 0.5) * CELL
            return math.hypot(x, z) <= NEAR_UNKNOWN_M
        return False

    cost = {(sx, sz): 0.0}
    parent = {}
    heap = [(0.0, sx, sz)]
    while heap:
        c, cx, cz = heapq.heappop(heap)
        if c > cost.get((cx, cz), INF):
            continue
        for dx, dz, w in neighbors8():
            nx_, nz_ = cx + dx, cz + dz
            if not inside(nx_, nz_) or not traversable(nx_, nz_):
                continue
            if dx != 0 and dz != 0 and not (traversable(cx + dx, cz) and traversable(cx, cz + dz)):
                continue  # no cutting corners
            pen = max(0.0, 3.0 - dist[nz_ * BEV_W + nx_]) * 0.5
            nc = c + w + pen
            if nc < cost.get((nx_, nz_), INF):
                cost[(nx_, nz_)] = nc
                parent[(nx_, nz_)] = (cx, cz)
                heapq.heappush(heap, (nc, nx_, nz_))

    gx, gz = goal_xz
    gcx, gcz = cell_of(gx, gz)
    gcx = min(max(gcx, 0), BEV_W - 1)
    gcz = min(max(gcz, 0), BEV_H - 1)
    if (gcx, gcz) in cost:
        end = (gcx, gcz)
    else:
        end = min(cost, key=lambda c: (math.hypot(c[0] - gcx, c[1] - gcz), cost[c]))
    path = [end]
    while path[-1] != (sx, sz):
        path.append(parent[path[-1]])
    path.reverse()

    # string pulling
    def clear_line(a, b):
        return all(traversable(x, z) for (x, z) in bresenham(a[0], a[1], b[0], b[1]))

    way = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not clear_line(path[i], path[j]):
            j -= 1
        way.append(path[j])
        i = j
    pts = [(X0 + (cx + 0.5) * CELL, (cz + 0.5) * CELL) for cx, cz in way]
    pts[0] = (0.0, 0.0)
    return pts, (end == (gcx, gcz))


def steer_bearing(path):
    need = LOOKAHEAD
    for k in range(len(path) - 1):
        (ax, az), (bx, bz) = path[k], path[k + 1]
        seg = math.hypot(bx - ax, bz - az)
        if seg >= need and seg > 1e-9:
            t = need / seg
            x, z = ax + (bx - ax) * t, az + (bz - az) * t
            return math.degrees(math.atan2(x, z))
        need -= seg
    x, z = path[-1]
    if math.hypot(x, z) < 0.3:
        return None
    return math.degrees(math.atan2(x, z))


def sectors(grid):
    out = {'left': None, 'center': None, 'right': None}
    for cz in range(BEV_H):
        for cx in range(BEV_W):
            if grid[cz * BEV_W + cx] not in (OBST, DROP):
                continue
            x = X0 + (cx + 0.5) * CELL
            z = (cz + 0.5) * CELL
            r = math.hypot(x, z)
            if r > 3.0 or abs(x) > 2.0:
                continue
            key = 'center' if abs(x) <= 0.5 else ('left' if x < 0 else 'right')
            if out[key] is None or r < out[key][0]:
                out[key] = (r, math.degrees(math.atan2(x, z)))
    return out


def ahead(grid):
    obst, free = None, None
    for cz in range(BEV_H):
        row_blocked = row_free = False
        for cx in range(BEV_W):
            x = X0 + (cx + 0.5) * CELL
            if abs(x) > 0.35:
                continue
            g = grid[cz * BEV_W + cx]
            if g in (OBST, DROP):
                row_blocked = True
            if g == FREE:
                row_free = True
        z = cz * CELL
        if row_blocked and obst is None:
            obst = z
        if not row_free and not row_blocked and free is None and z >= 1.0:
            free = z
    return obst, free


def open_bearing(grid, goal_bearing=0.0):
    """Bearing (deg) of the longest free run straight out from the user within +-30 deg, and that run's length."""
    best = (None, -1.0)
    for deg in range(-30, 31, 5):
        a = math.radians(deg)
        run = 0.0
        r = 0.2
        while r < BEV_H * CELL:
            cx, cz = cell_of(r * math.sin(a), r * math.cos(a))
            if not inside(cx, cz) or grid[cz * BEV_W + cx] != FREE:
                if not (r <= NEAR_UNKNOWN_M and inside(cx, cz) and grid[cz * BEV_W + cx] == UNKNOWN):
                    break
            run = r
            r += 0.2
        score = run - 0.002 * abs(deg - goal_bearing)  # tie-break toward the goal
        if score > best[1]:
            best = (float(deg), score)
    return best[0], max(best[1], 0.0)


class Planner:
    def __init__(self):
        self.rng = Rng()
        self.abc = None  # smoothed plane y = a x + b z + c
        self.abc_t = None

    @staticmethod
    def _plane_from_abc(a, b, c):
        lm = math.sqrt(a * a + 1.0 + b * b)
        n = (a / lm, -1.0 / lm, b / lm)
        d = -c / lm
        return n, d, -d

    def plan(self, depth, hfov_deg, goal=None, now_ms=0.0):
        """goal: (xn, yn, depth_m) = target box centre in normalised image coordinates and its depth, or None."""
        pts = unproject(depth, hfov_deg)
        fit = fit_ground(pts, self.rng)
        held = False
        if fit is not None:
            abc = fit[4]
            if self.abc is not None:
                abc = tuple(o + PLANE_ALPHA * (nw - o) for o, nw in zip(self.abc, abc))
            self.abc, self.abc_t = abc, now_ms
        elif self.abc is not None and now_ms - self.abc_t <= HOLD_MS:
            held = True  # not enough floor in view right now: keep the last good plane for a moment
        else:
            self.abc = None
            return {'ok': False, 'status': 'noGround'}
        n, d, h = self._plane_from_abc(*self.abc)
        grid, f_g, r_g = build_bev(pts, n, d)
        if goal is not None:
            xn, yn, zg = goal
            f = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
            x = (xn * W - W / 2) * zg / f
            y = (yn * H - H / 2) * zg / f
            gx = r_g[0] * x + r_g[1] * y + r_g[2] * zg
            gz = f_g[0] * x + f_g[1] * y + f_g[2] * zg
        else:
            gx, gz = 0.0, 4.0
        path, reached = plan_on_grid(grid, (gx, gz))
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
        progress = math.hypot(gx, gz) - math.hypot(path[-1][0] - gx, path[-1][1] - gz)
        blocked = (not reached) and (length < MIN_PATH_M or progress < MIN_PROGRESS_M)
        sb = None if blocked else steer_bearing(path)
        goal_bearing = math.degrees(math.atan2(gx, gz))
        ob, run = open_bearing(grid, goal_bearing)
        obst, free = ahead(grid)
        return {
            'ok': True, 'status': 'blocked' if blocked else 'ok', 'held': held,
            'h': h, 'pitch': math.degrees(math.atan2(-n[2], -n[1])),
            'steer': sb, 'reached': reached, 'path': path, 'sectors': sectors(grid),
            'obstacle_ahead': obst, 'free_ahead': free, 'open_bearing': ob, 'open_run': run, 'grid': grid,
        }


# ------------------------------------------------------------------ synthetic scenes
def render(hfov, cam_h, pitch_deg, boxes=(), wall_z=None, hole=None, noise=0.0, scale=1.0, seed=7):
    """Metric depth (Z along the camera axis) of a scene seen from height cam_h, pitched down by pitch_deg.
    boxes: (x0, x1, z0, z1, height) on the floor.  wall: plane z = wall_z.  hole: (x0, x1, z0, z1, depth) floor drop."""
    import numpy as np
    th = math.radians(pitch_deg)
    s, c = math.sin(th), math.cos(th)
    xc, yc, zc = np.array([1.0, 0, 0]), np.array([0.0, -c, -s]), np.array([0.0, -s, c])
    f = (W / 2) / math.tan(math.radians(hfov) / 2)
    O = np.array([0.0, cam_h, 0.0])
    rng = np.random.default_rng(seed)
    out = []
    for j in range(H):
        for i in range(W):
            dc = ((i + 0.5 - W / 2) / f, (j + 0.5 - H / 2) / f, 1.0)
            d = xc * dc[0] + yc * dc[1] + zc * dc[2]
            best = math.inf
            if d[1] < -1e-9:  # floor
                t = -O[1] / d[1]
                p = O + t * d
                if hole and hole[0] <= p[0] <= hole[1] and hole[2] <= p[2] <= hole[3]:
                    t2 = -(O[1] + hole[4]) / d[1]
                    p2 = O + t2 * d
                    if hole[0] <= p2[0] <= hole[1] and hole[2] <= p2[2] <= hole[3] and t2 > 0:
                        best = t2
                elif t > 0:
                    best = t
            if wall_z is not None and d[2] > 1e-9:
                t = (wall_z - O[2]) / d[2]
                if 0 < t < best:
                    best = t
            for (x0, x1, z0, z1, hh) in boxes:
                lo, hi = -math.inf, math.inf
                for k, (a0, a1) in enumerate(((x0, x1), (0.0, hh), (z0, z1))):
                    if abs(d[k]) < 1e-12:
                        if not (a0 <= O[k] <= a1):
                            lo, hi = 1, 0
                        continue
                    t0, t1 = (a0 - O[k]) / d[k], (a1 - O[k]) / d[k]
                    lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
                if lo <= hi and lo > 0 and lo < best:
                    best = lo
            z = best if best < math.inf else float('nan')
            if z == z:
                z = z * scale * (1 + noise * rng.standard_normal())
            out.append(z)
    return out


SCENES = {
    'open_floor': dict(args=dict(hfov=60, cam_h=1.3, pitch_deg=15), goal=None),
    'box_left': dict(args=dict(hfov=60, cam_h=1.3, pitch_deg=15, boxes=[(-0.9, -0.35, 1.8, 2.4, 0.8)]), goal=None),
    'box_right': dict(args=dict(hfov=60, cam_h=1.3, pitch_deg=15, boxes=[(0.35, 0.9, 1.8, 2.4, 0.8)]), goal=None),
    'wall_ahead_cold': dict(args=dict(hfov=60, cam_h=1.3, pitch_deg=15, wall_z=1.6), goal=None),
    'wall_ahead': dict(args=dict(hfov=60, cam_h=1.3, pitch_deg=15, wall_z=1.6), goal=None, after='open_floor'),
    'box_ahead_goal_behind_it': dict(args=dict(hfov=60, cam_h=1.3, pitch_deg=15, boxes=[(-0.4, 0.4, 2.2, 2.8, 0.9)]),
                                     goal=(0.5, 0.55, 4.2)),
    'drop_ahead': dict(args=dict(hfov=60, cam_h=1.3, pitch_deg=15, hole=(-1.5, 1.5, 2.6, 4.5, 0.6)), goal=None),
    'noisy_open_floor': dict(args=dict(hfov=60, cam_h=1.3, pitch_deg=15, noise=0.03, scale=1.1), goal=None),
    'very_noisy_open_floor': dict(args=dict(hfov=60, cam_h=1.3, pitch_deg=20, noise=0.06, scale=0.9, seed=11), goal=None),
    'noisy_drop': dict(args=dict(hfov=60, cam_h=1.3, pitch_deg=15, hole=(-1.5, 1.5, 2.6, 4.5, 0.6), noise=0.03),
                       goal=None),
    'noisy_box_left': dict(args=dict(hfov=60, cam_h=1.3, pitch_deg=15, boxes=[(-0.9, -0.35, 1.8, 2.4, 0.8)],
                                     noise=0.03, scale=1.1), goal=None),
}


def run(name):
    sc = SCENES[name]
    depth = render(**sc['args'])
    planner = Planner()
    if sc.get('after') is not None:  # warm the plane up on another scene first (the user walked up to the wall)
        planner.plan(render(**SCENES[sc['after']]['args']), sc['args']['hfov'], None, 0.0)
        return depth, planner.plan(depth, sc['args']['hfov'], sc['goal'], 500.0)
    return depth, planner.plan(depth, sc['args']['hfov'], sc['goal'], 0.0)


def show(grid, path=None):
    pset = set()
    if path:
        for (x, z) in path:
            pset.add(cell_of(x, z))
    for cz in range(BEV_H - 1, -1, -1):
        row = ''
        for cx in range(BEV_W):
            g = grid[cz * BEV_W + cx]
            ch = {UNKNOWN: '.', FREE: ' ', OBST: '#', DROP: 'v'}[g]
            if (cx, cz) in pset:
                ch = '*'
            if (cx, cz) == cell_of(0, 0):
                ch = '^'
            row += ch
        print(f'{cz * CELL:4.1f} |{row}|')


if __name__ == '__main__':
    import sys
    verbose = '-v' in sys.argv
    fixtures = pathlib.Path(__file__).resolve().parent.parent / 'test' / 'fixtures'
    fixtures.mkdir(parents=True, exist_ok=True)
    for name in SCENES:
        depth, r = run(name)
        sec = r.get('sectors') or {}
        fmt = lambda v: 'None' if v is None else f'{v[0]:.2f}m@{v[1]:+.0f}deg'
        print(f"{name:26s} ok={r['ok']!s:5} h={r.get('h', float('nan')):.2f} pitch={r.get('pitch', float('nan')):.1f} "
              f"steer={r.get('steer')!s:>22.22} reached={r.get('reached')} L={fmt(sec.get('left'))} "
              f"C={fmt(sec.get('center'))} R={fmt(sec.get('right'))} obstAhead={r.get('obstacle_ahead')}")
        if verbose and r['ok']:
            show(r['grid'], r['path'])
            print()
        if r['ok']:
            exp = {k: r[k] for k in ('status', 'h', 'pitch', 'steer', 'reached', 'obstacle_ahead', 'free_ahead',
                                     'open_bearing', 'held')}
            exp['sectors'] = {k: v for k, v in sec.items()}
            exp['path'] = r['path']
        else:
            exp = {'status': r['status']}
        sc = SCENES[name]
        (fixtures / f'{name}.json').write_text(json.dumps({
            'hfov': sc['args']['hfov'], 'goal': sc['goal'],
            'depth': [None if d != d else round(d, 5) for d in depth],
            'warmup_depth': ([None if d != d else round(d, 5) for d in render(**SCENES[sc['after']]['args'])]
                             if sc.get('after') else None),
            'expected': exp}))
    print('fixtures written to', fixtures)
