import time, random, os
from collections import defaultdict, namedtuple
import numpy as np
import matplotlib.pyplot as plt

Course = namedtuple('Course', 'name teacher cohort size duration')
Room   = namedtuple('Room',   'name capacity')

class TimetableInstance:
    def __init__(self, days=5, periods=6, rooms=None, courses=None, seed=None):
        if seed is not None: random.seed(seed); np.random.seed(seed)
        self.days, self.periods = days, periods
        self.T = days*periods  # linearize timeslots 0..T-1
        self.rooms = rooms if rooms is not None else self._default_rooms()
        self.courses = courses if courses is not None else self._random_courses()

    def _default_rooms(self):
        return [Room(f"R{i}", capacity) for i, capacity in enumerate([30, 40, 50, 60])]

    def _random_courses(self, n=18):
        teachers = [f"T{i}" for i in range(1, 9)]
        cohorts  = ["CSE-A","CSE-B","ECE-A","ECE-B"]
        courses = []
        for i in range(n):
            name = f"Sub{i+1}"
            teacher = random.choice(teachers)
            cohort = random.choice(cohorts)
            size = random.choice([25, 30, 35, 40, 45, 50])
            duration = random.choice([1,1,1,2])  # mostly single slot; some double
            courses.append(Course(name, teacher, cohort, size, duration))
        return courses

# --------------------------- CSP Core ---------------------------
class Metrics:
    def __init__(self):
        self.assignments = 0
        self.backtracks = 0
        self.constraint_checks = 0
        self.start = time.perf_counter()
        self.end = None
    @property
    def runtime_ms(self):
        return (self.end - self.start)*1000 if self.end else None

class TimetableCSP:
    def __init__(self, inst: TimetableInstance):
        self.I = inst
        # Domain for each course = all (room, start) that fit duration in horizon
        self.domain0 = {
            c: [(r_i, t) for r_i, r in enumerate(self.I.rooms)
                        for t in range(self.I.T - c.duration + 1)
                        if self.I.rooms[r_i].capacity >= c.size]
            for c in self.I.courses
        }

    # Constraint helpers
    def overlap(self, c: Course, t:
        int, c2: Course, t2: int):
        return not (t + c.duration <= t2 or t2 + c2.duration <= t)

    def check_constraints(self, partial_assign):
        # partial_assign: {Course: (room_index, start_t)}
        # pairwise check teacher/room/cohort overlaps
        items = list(partial_assign.items())
        ok = True
        for i in range(len(items)):
            c1,(r1,t1) = items[i]
            for j in range(i+1, len(items)):
                c2,(r2,t2) = items[j]
                # Teacher overlap
                self.metrics.constraint_checks += 1
                if c1.teacher == c2.teacher and self.overlap(c1,t1,c2,t2):
                    return False
                # Room overlap
                self.metrics.constraint_checks += 1
                if r1 == r2 and self.overlap(c1,t1,c2,t2):
                    return False
                # Cohort overlap
                self.metrics.constraint_checks += 1
                if c1.cohort == c2.cohort and self.overlap(c1,t1,c2,t2):
                    return False
        return ok

    # -------------------- Solvers --------------------
    def solve_backtracking(self, use_fc=False):
        self.metrics = Metrics()
        assign = {}
        domains = {c: list(self.domain0[c]) for c in self.I.courses}

        def select_var():
            # MRV, then Degree
            unassigned = [c for c in self.I.courses if c not in assign]
            unassigned.sort(key=lambda c: (len(domains[c]), -self.degree(c)))
            return unassigned[0]

        def order_values(c):
            # LCV: least-constraining value first
            vals = domains[c]
            scores = []
            for v in vals:
                score = 0
                for c2 in self.I.courses:
                    if c2 in assign or c2 == c: continue
                    score += self.value_conflicts(c, v, c2, domains[c2])
                scores.append((score, v))
            scores.sort(key=lambda x: x[0])
            return [v for _,v in scores]

        def backtrack():
            if len(assign) == len(self.I.courses):
                self.metrics.end = time.perf_counter()
                return True
            c = select_var()
            self.metrics.assignments += 1
            for v in order_values(c):
                assign[c] = v
                if self.check_constraints(assign):
                    # FC: prune future domains
                    pruned = []
                    if use_fc:
                        for c2 in self.I.courses:
                            if c2 in assign or c2 == c: continue
                            new_dom = []
                            for v2 in domains[c2]:
                                if self.consistent_pair(c, v, c2, v2):
                                    new_dom.append(v2)
                            if len(new_dom) < len(domains[c2]):
                                pruned.append((c2, domains[c2]))
                                domains[c2] = new_dom
                            if not domains[c2]:
                                # dead end
                                for cc, dom in reversed(pruned):
                                    domains[cc] = dom
                                del assign[c]
                                break
                        else:
                            if backtrack():
                                return True
                            # undo prunes on backtrack
                            for cc, dom in reversed(pruned):
                                domains[cc] = dom
                    else:
                        if backtrack():
                            return True
                # undo assign
                del assign[c]
                self.metrics.backtracks += 1
            if len(assign) == 0:
                self.metrics.end = time.perf_counter()
            return False

        ok = backtrack()
        return ok, assign if ok else None, self.metrics

    # -------------------- Heuristic helpers --------------------
    def degree(self, c):
        # number of other courses that can conflict by teacher/room/cohort
        deg = 0
        for c2 in self.I.courses:
            if c2 == c: continue
            if c.teacher == c2.teacher or c.cohort == c2.cohort:
                deg += 1
        # approximate room-sharing conflicts
        deg += len(self.I.courses) // max(1, len(self.I.rooms))
        return deg

    def value_conflicts(self, c, v, c2, dom2):
        # count values in dom2 that conflict with assigning c->v
        r1,t1 = v
        cnt = 0
        for r2,t2 in dom2:
            if not self.consistent_pair(c,(r1,t1),c2,(r2,t2)):
                cnt += 1
        return cnt

    def consistent_pair(self, c1, v1, c2, v2):
        r1,t1 = v1; r2,t2 = v2
        # teacher
        if c1.teacher == c2.teacher and self.overlap(c1,t1,c2,t2):
            return False
        # room
        if r1 == r2 and self.overlap(c1,t1,c2,t2):
            return False
        # cohort
        if c1.cohort == c2.cohort and self.overlap(c1,t1,c2,t2):
            return False
        return True

# --------------------------- Experiments ------------------------

def run_csp_trials(trials=20, seed=123):
    rng = random.Random(seed)
    rows = []
    for i in range(trials):
        inst = TimetableInstance(days=5, periods=6, seed=rng.randrange(10**9))
        csp = TimetableCSP(inst)
        ok_bt, sol_bt, m_bt = csp.solve_backtracking(use_fc=False)
        ok_fc, sol_fc, m_fc = csp.solve_backtracking(use_fc=True)
        rows.append({
            'bt_success': ok_bt,
            'bt_rt_ms': m_bt.runtime_ms,
            'bt_assign': m_bt.assignments,
            'bt_backtracks': m_bt.backtracks,
            'bt_checks': m_bt.constraint_checks,
            'fc_success': ok_fc,
            'fc_rt_ms': m_fc.runtime_ms,
            'fc_assign': m_fc.assignments,
            'fc_backtracks': m_fc.backtracks,
            'fc_checks': m_fc.constraint_checks,
        })
    return rows

def summarize(rows):
    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return (sum(vals)/len(vals)) if vals else float('nan')
    def rate(key):
        vals = [r[key] for r in rows]
        return sum(1 for v in vals if v)/len(vals)
    S = {
        'bt_success_rate': rate('bt_success'),
        'bt_rt_ms': avg('bt_rt_ms'),
        'bt_backtracks': avg('bt_backtracks'),
        'bt_checks': avg('bt_checks'),
        'fc_success_rate': rate('fc_success'),
        'fc_rt_ms': avg('fc_rt_ms'),
        'fc_backtracks': avg('fc_backtracks'),
        'fc_checks': avg('fc_checks'),
    }
    return S

def plot_csp(summary, out_png):
    os.makedirs('results', exist_ok=True)
    labels = ['success_rate','rt_ms','backtracks','checks']
    bt_vals = [summary['bt_success_rate'], summary['bt_rt_ms'], summary['bt_backtracks'], summary['bt_checks']]
    fc_vals = [summary['fc_success_rate'], summary['fc_rt_ms'], summary['fc_backtracks'], summary['fc_checks']]
    x = np.arange(len(labels))
    w = 0.35
    import matplotlib.pyplot as plt
    plt.figure()
    plt.bar(x - w/2, bt_vals, width=w, label='BT+Heur')
    plt.bar(x + w/2, fc_vals, width=w, label='BT+FC')
    plt.xticks(x, labels, rotation=15)
    plt.ylabel('value')
    plt.title('CSP methods comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

def show_timetable(assign, inst):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_title("Timetable Schedule")
    ax.set_xlabel("Period")
    ax.set_ylabel("Day")
    colors = plt.cm.tab10.colors
    for c, (r, t) in assign.items():
        day, period = divmod(t, inst.periods)
        ax.barh(day, c.duration, left=period, height=0.6,
                color=colors[r % len(colors)], edgecolor='black')
        ax.text(period+0.1, day, c.name, fontsize=8, va='center')
    ax.set_yticks(range(inst.days))
    ax.set_yticklabels([f"Day {d+1}" for d in range(inst.days)])
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    rows = run_csp_trials(trials=15, seed=7)
    S = summarize(rows)
    print("\nCSP Comparison (mean over runs):")
    for k,v in S.items():
        if isinstance(v, float):
            print(f"- {k:18s}: {v:.3f}")
        else:
            print(f"- {k:18s}: {v}")
    plot_csp(S, 'results/csp_comparison.png')
    inst = TimetableInstance(seed=10)
    csp = TimetableCSP(inst)
    ok, sol, m = csp.solve_backtracking(use_fc=True)
    if ok:
        show_timetable(sol, inst)
