## PatVis

Exposes a few easy interfaces to commonly used data visualization tools.  Currently, there is an interface for tSNE and network visualization libraries.

The main use of this code is for visualizing patent genealogies. There are several layout algorithms and coloring schemes available. The code allows one to specify
a patent number and the number of generations, from which it will generate a genealogy visualization by performing a breadth-first search against the database according to citation links.
The user may specify whether to follow only links into a patent or out of a patent (ancestors or descendants), or both.