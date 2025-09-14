## Voronoi Diagram and Fortune's Algorithm
A Voronoi diagram is a way of dividing space into regions based on distance to a set of given points. Each region contains all the locations that are closer to one specific point (called a site) than to any other. Voronoi diagrams appear in many fields, from computational geometry to geography and biology, because they naturally model concepts like influence zones and nearest-neighbor relationships.

To compute the diagram efficiently, I implemented Fortune’s algorithm, which is a plane sweep algorithm that constructs a Voronoi diagram in O(n log n) time. The idea is to move a horizontal sweep line from top to bottom across the plane and maintain a dynamic structure called the beach line. The beach line is made up of parabolic arcs, each defined by a site and the current position of the sweep line. At any moment, the beach line represents the boundary between the portion of the diagram already processed and the part that is yet to be explored.

The algorithm revolves around two types of events: site events and circle events. A site event occurs when the sweep line encounters a new site. At this moment, a new parabolic arc is added to the beach line, splitting an existing arc and creating new breakpoints that trace out Voronoi edges. A circle event happens when three consecutive arcs on the beach line converge to a single point, causing the middle arc to vanish. This convergence point becomes a Voronoi vertex, and the edges incident to it are finalized.

## Project Explanation
This project is a program that visualizes Fortune’s algorithm in action. While the algorithm itself works by handling events at discrete points (site events and circle events), the visualization makes the process easier to understand by showing how the beach line evolves continuously as the sweep line moves downward.

In practice, Fortune’s algorithm “jumps” the sweep line from one event to the next, since changes to the beach line only occur at those points. However, for visualization purposes, I implemented a way to smoothly adjust the sweep line and redraw the beach line at intermediate positions. This allows you to actually see the parabolic arcs grow, split, and disappear, rather than just witnessing sudden changes at event locations.

## Optimization
One challenge with visualizing Fortune’s algorithm is efficiently reconstructing the beach line at an arbitrary sweep line position. If we start from scratch—only given the positions of the sites—finding the beach line at a sweep line y requires running Fortune’s algorithm from the very top down to y. This takes O(n log n) time in the worst case, since the full algorithm must process all site and circle events up to that point.

To improve this, I designed a method to calculate the beach line at a given sweep line position in O(n) time, while keeping the space complexity at O(n). The key insight is to record how the beach line evolves over time in a compact way, instead of recomputing it from scratch.

I achieve this by storing the changes in the beach line as a **directed acyclic graph (DAG)**. Each node in the graph corresponds to an arc, and an edge between two nodes means that those arcs were neighbors at some point in the algorithm’s execution. Since each event only introduces a constant number of edges (at most 4 for a site event, 1 for a circle event), the total space is bounded by O(n).

When reconstructing the beach line at a sweep line position y, the program traverses this graph in order, guided by binary search, to find which arcs are active. As a result, the reconstruction runs in amortized O(j) time, where j is the number of arcs present in the beach line at that sweep position.

In summary, while the full Fortune’s algorithm runs in O(n log n) time and uses O(n) space, I optimized the visualization process to be more time-efficient. The program can visualize the construction of the Voronoi diagram in **O(n) time** in an **output-sensitive** manner, all while maintaining O(n) space complexity.