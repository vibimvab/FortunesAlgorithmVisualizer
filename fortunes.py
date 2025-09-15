from __future__ import annotations

from typing import Optional
import heapq
import math
import numpy as np


class FortunesAlgorithm:
    def __init__(self, points: list[tuple[float, float]]):
        self.points: set[tuple[float, float]] = set(points)
        self.event_queue = EventQueue(self.points)
        self.beachline = Beachline()
        self.voronoi_vertices = []
        self.ray_list: list[Ray] = []   # rays are stored in the order they are created

    def run(self):
        print("running fortune's algorithm with sites:")
        for site_event in self.event_queue.site_events:
            print(site_event.as_tuple())
        print()

        while not self.event_queue.is_empty():
            next_event = self.event_queue.pop()
            if isinstance(next_event, SiteEvent):
                self.handle_site_event(next_event)
            else:
                self.handle_circle_event(next_event)

        print("\nfinal ray list:")
        for ray in self.ray_list:
            print(ray.__repr__())
        print()

    def handle_site_event(self, event: SiteEvent):
        self.beachline.insert_arc(event, self.event_queue, self.ray_list)

    def handle_circle_event(self, next_event: CircleEvent):
        self.beachline.remove_arc(
            circle_event=next_event,
            event_queue=self.event_queue,
            voronoi_vertices=self.voronoi_vertices,
            ray_list=self.ray_list
        )

    def draw_rays(self, canvas, sweep_y: float):
        for ray in self.ray_list:
            if not ray.draw(canvas, sweep_y):
                break

    def draw_beachline(self, canvas, sweep_y: float):
        # beach line
        if sweep_y != float("inf"):
            self.beachline.draw(canvas, sweep_y)


class Event:
    def __init__(self, x, y):
        self.x: float = x   # x coordinate of the site
        self.y: float = y   # sweep line y that the event occurs

    def as_tuple(self):
        return self.x, self.y


class SiteEvent(Event):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __repr__(self):
        return f"SiteEvent({self.x}, {self.y})"

    def __eq__(self, other):
        if isinstance(other, SiteEvent):
            return self.x == other.x and self.y == other.y
        elif isinstance(other, CircleEvent):
            return False
        else:
            raise TypeError("Site event compared with non-event object")

    def __lt__(self, other):
        if isinstance(other, SiteEvent):
            return (self.y, self.x) < (other.y, other.x)
        elif isinstance(other, CircleEvent):
            # deliberately used <= so a site event is handled first when it has the same coordinate with a circle event
            return (self.y, self.x) <= (other.y, other.x)
        else:
            raise TypeError("Site event compared with non-event object")
        

class CircleEvent(Event):
    def __init__(self, x, y, center, arc):
        super().__init__(x, y)
        self.center = center        # Circle center (for Voronoi vertex)
        self.arc = arc              # The arc that will disappear
        self.valid = True           # To cancel false events

    def __repr__(self):
        return f"CircleEvent({self.x}, {self.y})"

    def __eq__(self, other):
        if isinstance(other, CircleEvent):
            return self.x == other.x and self.y == other.y
        elif isinstance(other, SiteEvent):
            return False
        else:
            raise TypeError("Circle event compared with non-event object")

    def __lt__(self, other):
        if isinstance(other, CircleEvent):
            return (self.y, self.x) < (other.y, other.x)
        elif isinstance(other, SiteEvent):
            return (self.y, self.x) < (other.y, other.x)
        else:
            raise TypeError("Circle event compared with non-event object")


class EventQueue:
    def __init__(self, sites: set[tuple[float, float]]):
        self.site_events: list[SiteEvent] = sorted([SiteEvent(x, y) for x, y in sites])
        self.site_index = 0
        self.circle_events: list[CircleEvent] = []

    def push(self, event: CircleEvent):
        if not isinstance(event, CircleEvent):
            raise TypeError("Event queue can only accept circle events")

        # add a circle event to the event queue
        heapq.heappush(self.circle_events, event)

    def pop(self) -> SiteEvent | CircleEvent:
        if self.is_empty():
            raise IndexError("Event queue is empty")

        # returns the event
        if not self.circle_events or (self.site_index < len(self.site_events) and self.site_events[self.site_index] < self.circle_events[0]):
            self.site_index += 1
            return self.site_events[self.site_index-1]
        else:
            circle_event = heapq.heappop(self.circle_events)
            if not circle_event.valid:
                return self.pop()
            else:
                return circle_event

    def is_empty(self):
        while self.circle_events and not self.circle_events[0].valid:
            heapq.heappop(self.circle_events)
        return not self.circle_events and not self.site_index < len(self.site_events)


class Ray:
    def __init__(self, start: tuple[float, float], left_site: SiteEvent, right_site: SiteEvent, start_sweep_y: float):
        self.start: tuple[float, float] = start
        self.end: Optional[tuple[float, float]] = None

        self.left_site: SiteEvent = left_site
        self.right_site: SiteEvent = right_site
        # self.twin: Optional[Ray] = None

        self.start_sweep_y = start_sweep_y
        self.end_sweep_y = None

    def __hash__(self):
        return hash(self.start + self.left_site.as_tuple())

    def __repr__(self):
        start_str = f"({self.start[0]:.3f}, {self.start[1]:.3f})"
        if self.is_trivial():
            return f"trivial ray at {start_str}, start sweep_y {self.start_sweep_y:.3f}"
        elif self.end:
            end_str = f"({self.end[0]:.3f}, {self.end[1]:.3f})"
            return f"Ray from {start_str} to {end_str}, start sweep_y {self.start_sweep_y:.3f}"
        else:
            return f"Ray from {start_str} in direction of {self.direction()}, start sweep_y {self.start_sweep_y:.3f}"

    def is_trivial(self):
        # start point and end point are the same
        if not self.end:
            return False
        else:
            return math.isclose(self.start[0], self.end[0], abs_tol=1e-9) and math.isclose(self.start[1], self.end[1], abs_tol=1e-9)

    def direction(self):
        # Returns the direction vector of the perpendicular bisector
        dx = self.right_site.x - self.left_site.x
        dy = self.right_site.y - self.left_site.y
        return dy, -dx  # since the y increases downward, this way we rotate the ray 90 degrees counterclockwise

    def find_intersection(self, other: Ray) -> Optional[tuple[float, float]]:
        # Get direction vectors and start points
        p1x, p1y = self.start
        d1x, d1y = self.direction()
        p2x, p2y = other.start
        d2x, d2y = other.direction()

        # Calculate determinant
        det = d1x * d2y - d1y * d2x
        if math.isclose(det, 0, abs_tol=1e-9):
            return None  # Lines are parallel

        # Calculate parameters for an intersection point
        t = ((p2x - p1x) * d2y - (p2y - p1y) * d2x) / det

        # Calculate intersection point
        x = p1x + t * d1x
        y = p1y + t * d1y

        return x, y

    def set_end(self, point: tuple[float, float], end_sweep_y: float):
        # Check if point lies on the ray's direction
        px, py = point
        sx, sy = self.start
        dx, dy = self.direction()

        # Cross-product to check alignment
        cross = dx * (py - sy) - (px - sx) * dy

        if not math.isclose(cross, 0, abs_tol=1e-9):
            raise ValueError("Point does not lie on the ray's direction")

        self.end = point
        self.end_sweep_y = end_sweep_y

    def orientation_test(self, point: tuple[float, float]) -> int:
        px, py = point
        sx, sy = self.start
        dx, dy = self.direction()

        # Cross product
        cross = dx * (py - sy) - (px - sx) * dy

        if math.isclose(cross, 0, abs_tol=1e-9):    # safeguard for floating point error
            return 0    # the point is on the edge
        elif cross > 0:
            return -1   # left
        else:
            return 1    # right

    def current_point(self, sweep_y: float) -> (float, float):
        """
        Return the current x position of this breakpoint for a sweep line sitting at y = sweep_y.
        """
        x1, y1 = self.left_site.as_tuple()
        x2, y2 = self.right_site.as_tuple()
        return ArcNode.compute_intersection(x1, y1, x2, y2, sweep_y)

    def draw(self, canvas, sweep_y: float) -> bool:
        """Draw the ray on the canvas"""
        # trivial case: the start point is the same with the end point (happens with degenerate cases)
        if self.is_trivial():
            return self.start_sweep_y < sweep_y or math.isclose(sweep_y, self.start_sweep_y, abs_tol=1e-9)

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # Case 1: beach line is above the ray -> pass
        if sweep_y < self.start_sweep_y or math.isclose(sweep_y, self.start_sweep_y, abs_tol=1e-9):
            return False

        x0, y0 = self.start

        # Case 2: finite ray, beach line is below the ray endpoint -> draw the full edge
        if self.end and (self.end_sweep_y < sweep_y or math.isclose(sweep_y, self.end_sweep_y, abs_tol=1e-9)):
            x1, y1 = self.end
            canvas.create_line(x0, y0, x1, y1, fill="black")
            return True

        # Case 3: half-infinite ray
        if not self.end:
            if sweep_y != float("inf"): # Case 3-1: "view" mode
                x1, y1 = self.current_point(sweep_y)
                if y1 == float("inf"):
                    canvas.create_line(x0, y0, x1, 0, fill="black")
                else:
                    y0 = max(0.0, y0)
                    canvas.create_line(x0, y0, x1, y1, fill="black")

            else:   # Case 3-2: "completed" mode, current_point doesn't work when sweep_y == inf
                # find the intersection between the half-infinite ray and the bounding box
                dx, dy = self.direction()
                if math.isclose(dx, 0, abs_tol=1e-9):
                    if dy > 0:
                        canvas.create_line(x0, y0, x0, canvas_height, fill="black")
                    else:
                        canvas.create_line(x0, y0, x0, 0, fill="black")
                elif dx > 0:
                    # check y value at x = canvas_width
                    y_end = y0 + (dy/dx) * (canvas_width - x0)
                    if y_end > canvas_height:
                        x_end = x0 + (dx/dy) * (canvas_height - y0)
                        canvas.create_line(x0, y0, x_end, canvas_height, fill="black")
                    elif y_end < 0:
                        x_end = x0 + (dx/dy) * (-y0)
                        canvas.create_line(x0, y0, x_end, 0, fill="black")
                    else:
                        canvas.create_line(x0, y0, canvas_width, y_end, fill="black")
                else:
                    # check y value at x = 0
                    y_end = y0 + (dy/dx) * (-x0)
                    if y_end > canvas_height:
                        x_end = x0 + (dx/dy) * (canvas_height - y0)
                        canvas.create_line(x0, y0, x_end, canvas_height, fill="black")
                    elif y_end < 0:
                        x_end = x0 + (dx/dy) * (-y0)
                        canvas.create_line(x0, y0, x_end, 0, fill="black")
                    else:
                        canvas.create_line(x0, y0, 0, y_end, fill="black")

            return True

        # Case 3: finite ray, beach line is below the ray endpoint -> draw the full edge
        if self.end_sweep_y < sweep_y or math.isclose(sweep_y, self.end_sweep_y, abs_tol=1e-9):
            x1, y1 = self.end
            canvas.create_line(x0, y0, x1, y1, fill="black")
            return True

        # Case 4: finite ray, beach line is above the ray endpoint -> find end point using sweep_y
        x1, y1 = self.current_point(sweep_y)
        if y1 == float("inf"):
            canvas.create_line(x0, y0, x1, 0, fill="black")
        else:
            canvas.create_line(x0, y0, x1, y1, fill="black")
        return True


class BeachlineNode:
    def __init__(self):
        pass


class BreakpointNode(BeachlineNode):
    def __init__(self, ray: Ray, parent: Optional[BreakpointNode], left: Optional[BreakpointNode | ArcNode] = None, right: Optional[BreakpointNode | ArcNode] = None, balance: int = 0):
        super().__init__()
        self.ray: Ray = ray                                 # ray the breakpoint is tracing
        # self.left_site: SiteEvent = ray.left_site
        # self.right_site: SiteEvent = ray.right_site
        self.parent: Optional[BreakpointNode] = parent      # parent node in the beachline tree, None if it is root
        self.left: BreakpointNode | ArcNode = left          # left child (breakpoint) node in the beachline tree
        self.right: BreakpointNode | ArcNode = right        # right child (breakpoint) node in the beachline tree
        self.bf: int = balance      # balance factor

    def get_bf(self):
        return self.bf

    def left_site(self) -> SiteEvent:
        return self.ray.left_site

    def right_site(self) -> SiteEvent:
        return self.ray.right_site

    def current_point(self, sweep_y: float) -> float:
        """Return the current position of this breakpoint for a sweep line sitting at y = sweep_y."""
        return self.ray.current_point(sweep_y)

    def get_child(self, point: tuple[float, float]) -> BreakpointNode | ArcNode:
        """For beachline traversal"""
        current_breakpoint_x, _ = self.current_point(point[1])    # find the current breakpoint position with respect to the sweep y
        if current_breakpoint_x > point[0]:
            return self.left
        else:
            return self.right

    def replace_arc(self, old_arc: ArcNode, new_bp: BreakpointNode):
        """replace arc node with breakpoint when the arc is split by a site event"""
        if self.left == old_arc:
            self.left = new_bp
        elif self.right == old_arc:
            self.right = new_bp
        else:
            raise ValueError("The breakpoint does not point to child being replaced")


class ArcNode(BeachlineNode):
    def __init__(self, site: SiteEvent, parent: BreakpointNode = None, circle_event: CircleEvent = None):
        super().__init__()
        self.site: SiteEvent = site                                 # site of the arc
        self.parent: Optional[BreakpointNode] = parent              # parent node in the beachline tree
        self.circle_event: Optional[CircleEvent] = circle_event     # circle event the arc disappears
        self.prev_arc: list[tuple[float, Optional[ArcNode]]] = []   # the arcs that had been on the left
        self.next_arc: list[tuple[float, Optional[ArcNode]]] = []   # the arcs that had been on the right

    def __repr__(self):
        return f"ArcNode(site={self.site.as_tuple()})"

    @staticmethod
    def get_bf():
        return 0

    def get_prev_arc_at(self, sweep_y) -> Optional[ArcNode]:
        """find the arc on the left when the sweep line is at y=sweep_y"""
        l = 0
        r = len(self.prev_arc) - 1
        result = -1
        while l <= r:
            mid = (l + r) // 2
            if self.prev_arc[mid][0] >= sweep_y:
                result = mid
                r = mid - 1
            else:
                l = mid + 1
        return self.prev_arc[result][1] if result != -1 else None

    def get_last_prev_arc(self) -> Optional[ArcNode]:
        """find the arc on the left"""
        return self.prev_arc[-1][1] if len(self.prev_arc) > 0 else None

    def get_next_arc_at(self, sweep_y) -> Optional[ArcNode]:
        """find the arc on the right when the sweep line is at y=sweep_y"""
        l = 0
        r = len(self.next_arc) - 1
        result = -1
        while l <= r:
            mid = (l + r) // 2
            if self.next_arc[mid][0] <= sweep_y:
                result = mid
                l = mid + 1
            else:
                r = mid - 1
        return self.next_arc[result][1] if result != -1 else None

    def get_last_next_arc(self) -> Optional[ArcNode]:
        """find the arc on the right"""
        return self.next_arc[-1][1] if len(self.next_arc) > 0 else None

    def append_prev_arc(self, sweep_y: float, arc: Optional[ArcNode]):
        """append arc to the left of the current arc"""
        self.prev_arc.append((sweep_y, arc))

    def append_next_arc(self, sweep_y: float, arc: Optional[ArcNode]):
        """append arc to the right of the current arc"""
        self.next_arc.append((sweep_y, arc))

    def get_left_breakpoint(self) -> Optional[BreakpointNode]:
        node = self
        while node.parent and node.parent.left == node:
            node = node.parent
        return node.parent if node.parent else None

    def get_right_breakpoint(self) -> Optional[BreakpointNode]:
        node = self
        while node.parent and node.parent.right == node:
            node = node.parent
        return node.parent if node.parent else None

    @staticmethod
    def compute_parabola_y(x, fx, fy, ly):
        if math.isclose(fy, ly, abs_tol=1e-9) or fy > ly:
            return float("inf")

        return ((x - fx)**2) / (2 * (fy - ly)) + (fy + ly) / 2

    def find_split_point(self, x: float, sweep_y: float) -> Optional[tuple[float, float]]:
        """
        Find the start point of the newly created ray supposing the arc is split by a site event at (x, sweep_y).
        """
        xi, yi = self.site.as_tuple()

        # Degenerate case 1: focus lies on the directrix -> return the midpoint of the focus and x
        if math.isclose(yi, sweep_y, abs_tol=1e-9):
            return (xi + x) / 2, float("-inf")

        # Degenerate case 2: focus lies below the directrix -> return None
        if yi > sweep_y:
            return None

        # General case
        return x, self.compute_parabola_y(x, xi, yi, sweep_y)

    @staticmethod
    def compute_intersection(x1, y1, x2, y2, ly) -> tuple[float, float]:
        """
        compute the intersection point of two parabolas sharing the same directrix

        :param x1: x coordinate of the left parabola's focus.
        :param y1: y coordinate of the left parabola's focus.
        :param x2: x coordinate of the right parabola's focus.
        :param y2: y coordinate of the right parabola's focus.
        :param ly: y coordinate of the directrix.
        :return: (x, y) of the intersection point.
        """

        # This function should not be accessed when at least one of the sites is below the sweep line
        if y1 > ly or y2 > ly:
            raise ValueError("One of the site is below the sweep line")

        # case 1: If sites have the same y-coordinate, breakpoint is their midpoint x-coordinate
        if math.isclose(y1, y2, abs_tol=1e-10):
            x = (x1 + x2) / 2
            y = ArcNode.compute_parabola_y(x, x1, y1, ly)
            return x, y

        # case 2: If one site is at sweep line, breakpoint is at that site
        if math.isclose(y1, ly, abs_tol=1e-10):
            return x1, ArcNode.compute_parabola_y(x1, x2, y2, ly)
        if math.isclose(y2, ly, abs_tol=1e-10):
            return x2, ArcNode.compute_parabola_y(x2, x1, y1, ly)

        # general case: find intersection by solving quadratic equation
        dy1 = y1 - ly
        dy2 = y2 - ly

        a = (1/dy1 - 1/dy2) / 2
        b = -x1/dy1 + x2/dy2
        c = (x1**2  + y1**2 - ly**2)/(2*dy1) - (x2**2  + y2**2 - ly**2)/(2*dy2)

        roots = np.roots([a, b, c])
        # if left site is below right site, break point is the intersection on the left, else the right one
        x = min(roots) if y1 > y2 else max(roots)
        y = ArcNode.compute_parabola_y(x, x1, y1, ly)

        return x, y


    def draw(self, canvas, sweep_y: float, left_x: float, step_size=1):
        """Draw the arc on the canvas"""

        # focus of this arc
        x1, y1 = self.site.as_tuple()
        if not math.isclose(sweep_y, y1, abs_tol=1e-9) and sweep_y < y1:
            raise ValueError("arc shouldn't be drawn when sweep line is below the site")

        # Base case: the arc is the last one in the beach line
        next_arc = self.get_next_arc_at(sweep_y)
        if not next_arc:
            if math.isclose(y1, sweep_y, abs_tol=1e-9): # the arc is the only one in the beach line
                canvas.create_line(x1, y1, x1, 0, fill="blue")

            else:   # draw the parabola to the right end of the bounding box
                sample_points = []
                for t in range(int(max(0.0, left_x)), canvas.winfo_width(), step_size):
                    x = t
                    y = self.compute_parabola_y(x, x1, y1, sweep_y)
                    sample_points.append((x, y))

                for i in range(len(sample_points) - 1):
                    x1, y1 = sample_points[i]
                    x2, y2 = sample_points[i + 1]
                    canvas.create_line(x1, y1, x2, y2, fill="blue", smooth=True)

            return float("inf")

        # focus of the next arc
        x2, y2 = next_arc.site.as_tuple()
        if not math.isclose(sweep_y, y2, abs_tol=1e-9) and sweep_y < y2:
            raise ValueError("arc shouldn't be drawn when sweep line is below the site")

        # focus in on the sweep line
        if math.isclose(y1, sweep_y, abs_tol=1e-9):
            if math.isclose(y2, sweep_y, abs_tol=1e-9):
                # focus of the next arc if also on the sweep line -> straight line up to the end
                canvas.create_line(x1, y1, x1, 0, fill="blue")

            else:
                # else -> straight line up to the next arc
                yb = ArcNode.compute_parabola_y(x1, x2, y2, sweep_y)
                canvas.create_line(x1, y1, x1, yb, fill="blue")

            return x1

        # the arc is on the left of the bounding box -> no need to draw
        xb, yb = self.compute_intersection(x2, y2, x1, y1, sweep_y) # xb, yb for breakpoint x, y
        if xb < 0:
            return xb

        # general case:
        sample_points = []
        for t in range(int(max(0.0, left_x)), int(min(xb, canvas.winfo_width())) + step_size, step_size):
            x = t
            y = self.compute_parabola_y(x, x1, y1, sweep_y)
            sample_points.append((x, y))

        for i in range(len(sample_points) - 1):
            x1, y1 = sample_points[i]
            x2, y2 = sample_points[i + 1]
            canvas.create_line(x1, y1, x2, y2, fill="blue", smooth=True)

        return xb


class Beachline:
    def __init__(self):
        self.root: Optional[BeachlineNode] = None
        self.leftmost_arc: Optional[list[tuple[float, ArcNode]]] = None

    def find_arc_above(self, site: SiteEvent) -> Optional[ArcNode]:
        """ Find the ArcNode vertically above the site event """
        if not self.root:    # if the beach line is empty, return None
            return None

        site_coordinate = site.as_tuple()
        current_node = self.root

        while isinstance(current_node, BreakpointNode):
            current_node = current_node.get_child(site_coordinate)

        assert(isinstance(current_node, ArcNode))
        return current_node

    def insert_arc(self, new_site: SiteEvent, event_queue: EventQueue, ray_list: list[Ray]):
        """Find the arc above new_site, insert the new arc, split edge, etc. """

        sweep_y = new_site.y
        print(f"handling site event {new_site.as_tuple()}")

        # Degenerate case 1: if the beachline is empty
        if not self.root:
            self.root = ArcNode(new_site)
            self.leftmost_arc = [(sweep_y, self.root)]
            return

        # Step 1: Find the arc above new_site
        arc_above = self.find_arc_above(new_site)

        # Step 2: Invalidate circle event of the arc being split
        if arc_above.circle_event:
            arc_above.circle_event.valid = False
            arc_above.circle_event = None

        # Degenerate case 2: if the site of the arc above has the same y with the new site's y
        if math.isclose(arc_above.site.y, new_site.y):
            # Step 3: Create new arcs
            left_arc = ArcNode(arc_above.site)
            right_arc = ArcNode(new_site)

            # Step 4: Create new rays (bisectors)
            ray_start = arc_above.find_split_point(new_site.x, sweep_y)
            # the new site is always on the right since the left one is handled first
            ray_down = Ray(start=ray_start, left_site=new_site, right_site=arc_above.site, start_sweep_y=sweep_y)
            ray_list.append(ray_down)

            # Step 5: Create breakpoints and hook up child relationships
            bp = BreakpointNode(ray=ray_down, parent=arc_above.parent, left=left_arc, right=right_arc, balance=0)
            left_arc.parent = bp
            right_arc.parent = bp

            # Step 6: update prev/next arc
            # prev_arc <-> left_arc
            prev_arc = arc_above.get_last_prev_arc()
            if prev_arc:
                prev_arc.next_arc.append((sweep_y, left_arc))
            else:
                self.leftmost_arc.append((sweep_y, left_arc))
            left_arc.append_prev_arc(sweep_y, prev_arc)

            # left_arc <-> right_arc
            left_arc.append_next_arc(sweep_y, right_arc)
            right_arc.append_prev_arc(sweep_y, left_arc)

            # right_arc <-> next_arc
            next_arc = arc_above.get_last_next_arc()
            if next_arc:
                right_arc.append_next_arc(sweep_y, next_arc)
                next_arc.append_prev_arc(sweep_y, right_arc)

            # Step 7: make the parent node point the new breakpoint subtree
            if bp.parent is None:
                self.root = bp
            elif bp.parent.left == arc_above:
                bp.parent.left = bp
            elif bp.parent.right == arc_above:
                bp.parent.right = bp
            else:
                raise RuntimeError("Parent does not point to child being replaced")

            # Circle event cannot appear in this case
            return

        # Step 3: Create new arcs
        left_arc: ArcNode = ArcNode(arc_above.site)
        center_arc: ArcNode = ArcNode(new_site)
        right_arc: ArcNode = ArcNode(arc_above.site)

        # Step 4: Create new rays (bisectors)
        ray_start = arc_above.find_split_point(new_site.x, sweep_y)
        ray_left = Ray(start=ray_start, left_site=new_site, right_site=arc_above.site, start_sweep_y=sweep_y)
        ray_right = Ray(start=ray_start, left_site=arc_above.site, right_site=new_site, start_sweep_y=sweep_y)
        ray_list.append(ray_left)   # append newly created rays to the ray list. the rays are stored in the order they are created
        ray_list.append(ray_right)

        # Step 5: Create breakpoints and hook up child relationships
        bp_left = BreakpointNode(ray=ray_left, parent=arc_above.parent, left=left_arc, balance=1)
        if self.root == arc_above:
            self.root = bp_left
        bp_right = BreakpointNode(ray=ray_right, parent=bp_left, left=center_arc, right=right_arc, balance=0)
        bp_left.right = bp_right

        left_arc.parent = bp_left
        center_arc.parent = bp_right
        right_arc.parent = bp_right

        # Step 6: update prev/next arc
        # prev_arc <-> left_arc
        prev_arc = arc_above.get_last_prev_arc()
        if prev_arc:
            prev_arc.next_arc.append((sweep_y, left_arc))
        else:
            self.leftmost_arc.append((sweep_y, left_arc))
        left_arc.append_prev_arc(sweep_y, prev_arc)

        # left_arc <-> center_arc
        left_arc.append_next_arc(sweep_y, center_arc)
        center_arc.append_prev_arc(sweep_y, left_arc)

        # center_arc <-> right_arc
        center_arc.append_next_arc(sweep_y, right_arc)
        right_arc.append_prev_arc(sweep_y, center_arc)

        # right_arc <-> next_arc
        next_arc = arc_above.get_last_next_arc()
        if next_arc:
            right_arc.append_next_arc(sweep_y, next_arc)
            next_arc.append_prev_arc(sweep_y, right_arc)

        # Step 7: make the parent node point the new breakpoint subtree
        parent = arc_above.parent
        if parent is None:
            self.root = bp_left
        elif parent.left == arc_above:
            parent.left = bp_left
        elif parent.right == arc_above:
            parent.right = bp_left
        else:
            raise RuntimeError("Parent does not point to child being replaced")

        # todo: rebalance the tree
        # parent_node = bp_left.parent
        # parent_node.bf += 2
        # self._balance(parent_node)

        # Step 8: create circle events
        left_left_arc = left_arc.get_last_prev_arc()
        if left_left_arc:
            left_event = self.check_circle_event(left_left_arc, left_arc, center_arc, sweep_y=new_site.y)
            if left_event:
                event_queue.push(left_event)
                # print("circle event created (site event, left)")

        right_right_arc = right_arc.get_last_next_arc()
        if right_right_arc:
            right_event = self.check_circle_event(center_arc, right_arc, right_right_arc, sweep_y=new_site.y)
            if right_event:
                event_queue.push(right_event)
                # print("circle event created (site event, right)")

        # print(f"current ray status:")
        # for ray in ray_list:
        #     print(ray)

        return

    @staticmethod
    def check_circle_event(left_arc: ArcNode, center_arc: ArcNode, right_arc: ArcNode, sweep_y: float) -> Optional[CircleEvent]:
        """
        Check whether a circle event should be generated for the given arc.
        If so, return the newly created CircleEvent.
        """

        def ccw(_p1, _p2, _p3) -> float:
            """Returns >0 when counter-clockwise, <0 when clockwise, 0 when collinear"""
            # because the y value increases downward, the positive determinant indicates visual clockwise contrary to the convention
            (x1, y1), (x2, y2), (x3, y3) = _p1, _p2, _p3
            det = (y2 - y1) * (x3 - x2) - (x2 - x1) * (y3 - y2)
            return 0 if math.isclose(det, 0, abs_tol=1e-9) else det

        def compute_circle(_p1, _p2, _p3) -> Optional[tuple[float, float], float]:
            """Compute the circle passing through 3 points. Return center and radius of the circle."""
            (x1, y1), (x2, y2), (x3, y3) = _p1, _p2, _p3

            a = x1 - x2
            b = y1 - y2
            c = x1 - x3
            d = y1 - y3
            e = (x1 ** 2 - x2 ** 2 + y1 ** 2 - y2 ** 2) / 2
            f = (x1 ** 2 - x3 ** 2 + y1 ** 2 - y3 ** 2) / 2

            # a * cx + b * cy = e
            # c * cx + d * cy = f

            det = a * d - b * c
            if math.isclose(det, 0, abs_tol=1e-9):
                return None     # Three points collinear

            cx = (d * e - b * f) / det
            cy = (-c * e + a * f) / det
            r = math.hypot(cx - x1, cy - y1)
            return (cx, cy), r

        # create a new circle event involving arc
        p1 = left_arc.site.as_tuple()
        p2 = center_arc.site.as_tuple()
        p3 = right_arc.site.as_tuple()

        # three points bend outward (counter-clockwise or collinear), return None
        if ccw(p1, p2, p3) >= 0:
            return None

        # compute the circle
        result = compute_circle(p1, p2, p3)
        if result is None:
            return None

        center, radius = result
        bottom_y = center[1] + radius

        # circle event lies above the sweep line
        if bottom_y < sweep_y and not math.isclose(bottom_y, sweep_y, abs_tol=1e-9):
            return None

        event = CircleEvent(center_arc.site.x, bottom_y, center=center, arc=center_arc)
        center_arc.circle_event = event
        return event

    def remove_arc(self, circle_event: CircleEvent, event_queue: EventQueue, voronoi_vertices: list[tuple[float, float]], ray_list: list[Ray]):
        arc: ArcNode = circle_event.arc
        if not arc:
            raise ValueError("The circle event does not belong to any arc")

        sweep_y = circle_event.y
        print(f"handling circle event at y={sweep_y}")

        # Get neighbors of the disappearing arc
        left_arc = arc.get_last_prev_arc()
        right_arc = arc.get_last_next_arc()

        # Indicate that this arc is no longer valid after this sweep_y
        arc.append_next_arc(sweep_y, None)
        arc.append_prev_arc(sweep_y, None)


        if not left_arc or not right_arc:
            raise ValueError("The arc cannot disappear with circle event when it does not have both neighbors")

        # Invalidate any circle events of neighboring arcs
        if left_arc.circle_event:
            left_arc.circle_event.valid = False
            left_arc.circle_event = None
        if right_arc.circle_event:
            right_arc.circle_event.valid = False
            right_arc.circle_event = None

        # Create a new ray and update Voronoi vertex
        new_ray = Ray(
            start=circle_event.center,
            left_site=right_arc.site,
            right_site=left_arc.site,
            start_sweep_y=sweep_y
        )
        voronoi_vertices.append(circle_event.center)
        ray_list.append(new_ray)

        # Set ends of old rays
        left_bp = arc.get_left_breakpoint()
        right_bp = arc.get_right_breakpoint()

        if not left_bp or not right_bp:
            raise ValueError("The arc cannot disappear with circle event when it does not have both breakpoints")

        left_bp.ray.set_end(circle_event.center, sweep_y)
        right_bp.ray.set_end(circle_event.center, sweep_y)

        # ---- Tree restructuring ----
        # Either one of left_bp or right_bp is always the parent of the arc to remove
        if arc.parent == left_bp:
            # arc is under left_bp, so replace right_bp with new_bp and remove left_bp
            new_bp = BreakpointNode(new_ray, parent=right_bp.parent)

            # replace the right_bp with new_bp
            # copy parent relationship
            if right_bp.parent:
                if right_bp.parent.left == right_bp:
                    right_bp.parent.left = new_bp
                else:
                    right_bp.parent.right = new_bp
            else:
                self.root = new_bp

            # copy child relationships
            new_bp.left = right_bp.left
            new_bp.right = right_bp.right
            if new_bp.left:
                new_bp.left.parent = new_bp
            if new_bp.right:
                new_bp.right.parent = new_bp

            # remove left_bp and the arc from the tree
            if not left_bp.parent:
                raise AssertionError("left_bp must have a parent")

            if left_bp == left_bp.parent.left:
                left_bp.parent.left = left_bp.left
            else:
                left_bp.parent.right = left_bp.left
            if left_bp.left:
                left_bp.left.parent = left_bp.parent

        elif arc.parent == right_bp:
            # arc is under right_bp, so replace left_bp with new_bp and remove right_bp
            new_bp = BreakpointNode(new_ray, parent=left_bp.parent)

            # replace the left_bp with new_bp
            # copy parent relationship
            if left_bp.parent:
                if left_bp.parent.left == left_bp:
                    left_bp.parent.left = new_bp
                else:
                    left_bp.parent.right = new_bp
            else:
                self.root = new_bp

            # copy child relationships
            new_bp.left = left_bp.left
            new_bp.right = left_bp.right
            if new_bp.left:
                new_bp.left.parent = new_bp
            if new_bp.right:
                new_bp.right.parent = new_bp

            # remove right_bp and the arc from the tree
            if not right_bp.parent:
                raise AssertionError("right_bp must have a parent")

            if right_bp == right_bp.parent.left:
                right_bp.parent.left = right_bp.right
            else:
                right_bp.parent.right = right_bp.right
            if right_bp.right:
                right_bp.right.parent = right_bp.parent

        else:
            raise ValueError("The arc to remove does not belong to the left or right breakpoint")

        # Update arc connectivity
        left_arc.append_next_arc(sweep_y, right_arc)
        right_arc.append_prev_arc(sweep_y, left_arc)

        # Create new circle events
        left_left = left_arc.get_last_prev_arc()
        if left_left:
            left_event = self.check_circle_event(left_left, left_arc, right_arc, sweep_y)
            if left_event:
                event_queue.push(left_event)
                # print("circle event created (circle event, left)")

        right_right = right_arc.get_last_next_arc()
        if right_right:
            right_event = self.check_circle_event(left_arc, right_arc, right_right, sweep_y)
            if right_event:
                event_queue.push(right_event)
                # print("circle event created (circle event, right)")

        # print(f"current ray status:")
        # for ray in ray_list:
        #     print(ray)

    def find_first_arc_at(self, sweep_y):
        """find the first arc when the sweep line is at y=sweep_y"""
        l = 0
        r = len(self.leftmost_arc) - 1
        result = -1
        while l <= r:
            mid = (l + r) // 2
            if self.leftmost_arc[mid][0] <= sweep_y:
                result = mid
                l = mid + 1
            else:
                r = mid - 1
        return self.leftmost_arc[result][1] if result != -1 else None

    def draw(self, canvas, sweep_y: float):
        """Draw the beachline on the canvas"""
        arc = self.find_first_arc_at(sweep_y) if self.leftmost_arc else None
        current_x = float("-inf")
        while arc and current_x < canvas.winfo_width(): # stop when the next arc is outside the bounding box
            current_x = arc.draw(canvas, sweep_y, current_x)
            arc = arc.get_next_arc_at(sweep_y)

    def print_current_arcs(self, sweep_y: float):
        """Print the current arcs when the sweep line is at y=sweep_y"""
        arc = self.find_first_arc_at(sweep_y) if self.leftmost_arc else None
        while arc:
            print(arc)
            arc = arc.get_next_arc_at(sweep_y)
        print()


    # private methods for self-balancing
    def _balance(self, current: Optional[BreakpointNode]):
        if not current:
            return current

        if current.get_bf() == -2:
            # if the node is left-heavy
            if current.left.get_bf() == -1 or current.left.get_bf() == 0:
                # left-left case
                return self._balance_left(current)
            elif current.left.get_bf() == 1:
                # left-right case
                return self._balance_left_right(current)
            else:
                raise ValueError("the beach line tree is not balanced")

        elif current.get_bf() == 2:
            # if the node is right-heavy
            if current.right.get_bf() == -1:
                # right-left case
                return self._balance_right_left(current)
            elif current.right.get_bf() == 1 or current.right.get_bf() == 0:
                # right-right case
                return self._balance_right(current)
            else:
                raise ValueError("the beach line tree is not balanced")

        elif -2 < current.get_bf() < 2:
            return current

        else:
            raise ValueError("the beach line tree is not balanced")


    def _balance_left(self, current: BreakpointNode):
        child = current.left
        assert isinstance(child, BreakpointNode)

        # left-left shift
        current.left = child.right
        child.right = current

        # set parent relationships
        child.parent = current.parent
        if not child.parent:
            self.root = child
        current.parent = child

        current.bf = 0
        child.bf = 0

    def _balance_left_right(self, current: BreakpointNode):
        child = current.left
        grand = child.right
        assert isinstance(child, BreakpointNode)
        assert isinstance(grand, BreakpointNode)
        grand_bf = grand.get_bf()

        # make it left-left
        child.right = grand.left
        grand.left = child
        current.left = grand

        # left-left shift
        current.left = grand.right
        grand.right = current

        # set parent relationships
        grand.parent = current.parent
        if not grand.parent:
            self.root = grand
        current.parent = grand
        child.parent = grand

        # adjust balance factors
        grand.bf = 0
        if grand_bf > 0:
            current.bf = 0
            child.bf = -1
        elif grand_bf < 0:
            current.bf = 1
            child.bf = 0
        else:
            current.bf = 0
            child.bf = 0

        return grand

    def _balance_right(self, current: BreakpointNode):
        child = current.right
        assert isinstance(child, BreakpointNode)

        # right-right shift
        current.right = child.left
        child.left = current

        # set parent relationships
        child.parent = current.parent
        if not child.parent:
            self.root = child
        current.parent = child

        current.bf = 0
        child.bf = 0

    def _balance_right_left(self, current: BreakpointNode):
        child = current.right
        grand = child.left
        assert isinstance(child, BreakpointNode)
        assert isinstance(grand, BreakpointNode)
        grand_bf = grand.get_bf()

        # make it right-right
        child.left = grand.right
        grand.right = child
        current.right = grand

        # right-right shift
        current.right = grand.left
        grand.left = current

        # set parent relationships
        grand.parent = current.parent
        if not grand.parent:
            self.root = grand
        current.parent = grand
        child.parent = grand

        # adjust balance factors
        grand.bf = 0
        if grand_bf < 0:
            current.bf = 0
            child.bf = 1
        elif grand_bf > 0:
            current.bf = -1
            child.bf = 0
        else:
            current.bf = 0
            child.bf = 0

        return grand


if __name__ == "__main__":
    input_sets = [[(0, 0), (2, 0), (2, 2)],
                  [(0, 0), (2, 0), (2, 2), (0, 2)],
                  [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)],
                  [(0, 0), (2, 0), (4, 0)]]

    for inputs in input_sets:
        fa = FortunesAlgorithm(inputs)
        fa.run()
        output_vertices = fa.voronoi_vertices
        print(output_vertices)
        print(fa.ray_list)
        print()
