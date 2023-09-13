//+
Point(1) = {0, 0.2, 0, 1.0};
//+
Point(2) = {0, 1, 0, 1.0};
//+
Point(3) = {0, -1, 0, 1.0};
//+
Point(4) = {0, -0.2, 0, 1.0};
//+
Point(5) = {0, 0, 0, 1.0};
//+
Circle(1) = {4, 5, 1};
//+
Circle(2) = {3, 5, 2};
//+
Line(3) = {1, 2};
//+
Line(4) = {3, 4};
//+
Curve Loop(1) = {1, 3, -2, 4};
//+
Surface(1) = {1};
//+
Physical Surface("domain") = {1};
//+
Physical Curve("boundary") = {3, 2, 4, 1};
