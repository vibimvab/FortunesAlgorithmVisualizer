## Voronoi Diagram and Fortune's Algorithm
A Voronoi diagram is a way of dividing space into regions based on distance to a set of given points. Each region contains all the locations that are closer to one specific point (called a site) than to any other. Voronoi diagrams appear in many fields, from computational geometry to geography and biology, because they naturally model concepts like influence zones and nearest-neighbor relationships.

To compute the diagram efficiently, I implemented Fortune’s algorithm, which is a plane sweep algorithm that constructs a Voronoi diagram in O(n log n) time. The idea is to move a horizontal sweep line from top to bottom across the plane and maintain a dynamic structure called the beach line. The beach line is made up of parabolic arcs, each defined by a site and the current position of the sweep line. At any moment, the beach line represents the boundary between the portion of the diagram already processed and the part that is yet to be explored.

The algorithm revolves around two types of events: site events and circle events. A site event occurs when the sweep line encounters a new site. At this moment, a new parabolic arc is added to the beach line, splitting an existing arc and creating new breakpoints that trace out Voronoi edges. A circle event happens when three consecutive arcs on the beach line converge to a single point, causing the middle arc to vanish. This convergence point becomes a Voronoi vertex, and the edges incident to it are finalized.

## Project Explanation
This project is a program that visualizes Fortune’s algorithm in action. While the algorithm itself works by handling events at discrete points (site events and circle events), the visualization makes the process easier to understand by showing how the beach line evolves continuously as the sweep line moves downward.

In practice, Fortune’s algorithm “jumps” the sweep line from one event to the next, since changes to the beach line only occur at those points. However, for visualization purposes, I implemented a way to smoothly adjust the sweep line and redraw the beach line at intermediate positions. This allows you to actually see the parabolic arcs grow, split, and disappear, rather than just witnessing sudden changes at event locations.

## Space-Time Trade-off
