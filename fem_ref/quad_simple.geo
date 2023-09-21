
params0 = 0;
params1 = 0;
params2 = 0;
params3 = 0;
meshsize = 0.002;

Do = 72e-3;                                                            
Di = 51e-3;                                                   
hi = 13e-3;                                                
bli = 3e-3;                                                             
Dc = 3.27640e-2;                                                   
hc = 7.55176e-3;                                                           
ri = 20e-3;                                                    
ra = 18e-3;                                                           
blc = hi-hc; 
rm = ((Dc+params1)*(Dc+params1)+hc*hc-(ri+params0)*(ri+params0))/((Dc+params1)*Sqrt(2)+hc*Sqrt(2)-2*(ri+params0));        
R = rm-ri;
O = rm/Sqrt(2);

Point(0) = {0,0,0,meshsize};
Point(1) = {Dc+params1,0,0,meshsize};
Point(2) = {Di+params2,0,0,meshsize/5};
Point(3) = {Do,0,0,meshsize};
Point(4) = {(ri+params0)/Sqrt(2),(params0+ri)/Sqrt(2),0,meshsize/5};
Point(5) = {Dc+params1,hc,0,meshsize/5};
Point(6) = {Di+params2,hi-bli,0,meshsize/5};
Point(7) = {Do,Do*Tan(Pi/8),0,meshsize};
Point(8) = {Dc+blc,hi+params3,0,meshsize/5};
Point(9) = {Di-bli,hi+params3,0,meshsize/5};
Point(10) = {O,O,0,meshsize};
Point(11) = {Do/Sqrt(2),Do/Sqrt(2),0,meshsize};

//+
Line(1) = {0, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 7};
//+
Line(5) = {11, 7};
//+
Line(6) = {10, 4};
//+
Line(7) = {0, 4};
//+
Line(8) = {1, 5};
//+
Line(9) = {8, 5};
//+
Line(10) = {8, 9};
//+
Line(11) = {9, 6};
//+
Line(12) = {2, 6};
//+
Line(13) = {10, 11};
//+
Ellipse(14) = {4, 10, 10, 5};
//+
Curve Loop(1) = {14, -8, -1, 7};
//+
Surface(1) = {1};
//+
Curve Loop(2) = {13, 5, -4, -3, 12, -11, -10, 9, -14, -6};
//+
Plane Surface(2) = {2};
//+
Physical Curve("GammaN", 15) = {1, 2, 3};
//+
Physical Curve("GammaD", 16) = {6, 13, 5, 4, 7};
//+
Physical Surface("Iron", 1) = {2};
//+
Physical Surface("Air", 2) = {1};
//+
Curve Loop(3) = {10, 11, -12, -2, 8, -9};
//+
Plane Surface(3) = {3};
//+
Physical Surface("Cu", 3) = {3};

// Point(10) = {0.032764000000000004231,0.007551760000000000417, 0, meshsize};
// Point(11) = {0.038212240000000002871,0.013000000000000000444, 0, meshsize};
// Point(12) = {0.047999999999999992673,0.013000000000000000444, 0, meshsize};
// Point(13) = {0.051000000000000000888,0.009999999999999997780, 0, meshsize};
// Point(14) = {0.051000000000000000888,0.000000000000000000000, 0, meshsize};
// 
// Point(20) = {0.071999999999999997335,0.000000000000000000000, 0, meshsize};
// Point(21) = {0.071999999999999997335,0.029823376490862840704, 0, meshsize};
// Point(22) = {0.050911688245431419020,0.050911688245431419020, 0, meshsize};
// Point(23) = {0.014142135623730950345,0.014142135623730950345, 0, meshsize};
// 
// Point(30) = {0.032764000000000004231,0.000000000000000000000, 0, meshsize};
// Point(31) = {0.051000000000000000888,0.000000000000000000000, 0, meshsize};
// 
// Point(40) = {0.000000000000000000000,0.000000000000000000000, 0, meshsize};
// Point(41) = {0.014142135623730950345,0.014142135623730950345, 0, meshsize};
// 
// Point(50) = {0.000000000000000000000,0.000000000000000000000, 0, meshsize};
// Point(51) = {0.032764000000000004231,0.000000000000000000000, 0, meshsize};
// 
// Point(60) = {0.030358203798490668301,0.030358203798490668301, 0, meshsize};
// 
// Point(70) = {0.032764000000000004231,0.000000000000000000000, 0, meshsize};
// Point(71) = {0.032764000000000004231,0.007551760000000000417, 0, meshsize};
// 
// Point(80) = {0.051000000000000000888,0.000000000000000000000, 0, meshsize};
// Point(81) = {0.071999999999999997335,0.000000000000000000000, 0, meshsize};
// 
// 
// 
// //+
// Line(1) = {40, 23};
// //+
// Line(2) = {22, 21};
// //+
// Line(3) = {20, 21};
// //+
// Line(4) = {60, 22};
// //+
// Line(5) = {23, 60};
// //+
// Line(6) = {40, 30};
// //+
// Line(7) = {30, 10};
// //+
// Line(8) = {11, 10};
// //+
// Line(9) = {11, 12};
// //+
// Line(10) = {13, 12};
// //+
// Line(11) = {14, 13};
// //+
// Line(12) = {14, 20};
// //+
// Line(13) = {30, 14};
// //+
// Circle(14) = {23, 60, 10};
// //+
// Line Loop(1) = {1, 14, -7, -6};
// //+
// Plane Surface(1) = {1};
// //+
// Line Loop(2) = {7, -8, 9, -10, -11, -13};
// //+
// Plane Surface(2) = {2};
// //+
// Line Loop(3) = {14, -8, 9, -10, -11, 12, 3, -2, -4, -5};
// //+
// Plane Surface(3) = {3};
// 
// //+
// Physical Line(1) = {1, 5, 4, 2, 3};
// //+
// Physical Surface(1) = {3};
// //+
// Physical Surface(2) = {1};
// //+
// Physical Surface(3) = {2};
// 