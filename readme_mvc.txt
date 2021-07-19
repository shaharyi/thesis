Matlab symbolic integration:
    syms a b x
    f = 3*a*b*x + 2*exp(a*x + b);
    int(f,x) 

Idea:
1. contintue each line with "border-color" upto the border.
2. find convex-hull, draw it in "border-color" 
3. use flood-fill for each pixel, unless already labeled.
  when reaching segment-pixels, remember for each segment it's max boundaries.
  So for each flooded "cell", we have set of small segments with their "color"(value).
4. for this cell, flood it again or use its remembered runs, to calculate for each pixel
the mvc.


LOG:

The lines now continue till border.
But there are 2 points in the transition from border to BG of another color.
This causes 2 border segments instead of 1 at line start.
Note that seg-val=0 for border segment.

16/4/2011
no need in "extra" segments, just the convex hull with border-color.
redraw:
each segment is assigned a color (id).
segment crossing is colored with special color (no id).

start with segment 1, first point, flood-search empty pt, only convexhull is border.
put in "Cell candidate points que"

pop and verify candidate is unassigned.
flood-fill (border is convhull, segments, crossing), record segment borders along with its watermarks.
for segment pt, insert unassigned pt across it into the que.
every flooded pt is put is cell-list.
when done flood-fill, process cell-list to compute MVC.
repeat this paragraph until que is empty.

17/4/2011
No need to consider crossing!
don't use special color for it.
lines may not have cross-point sometimes.

TODO
Review the color-coding system, to allow more than 255 segments.
