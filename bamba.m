N=185;


%Sphere of 1s with radius 25 centered at 50,50,50 in a 100x100x100 zero matrix
%%%
%[xx yy zz] = meshgrid(1:256,1:256,1:256);
%S = sqrt((xx-128).^2+(yy-128).^2+(zz-128).^2)<=32 || sqrt((xx-50).^2+(yy-50).^2+(zz-50).^2)<=12;

%Visualize
%isosurface(S,0)


% Centers=[-0.1 -0.2 -0.5;
% 0.2 0.7 -0.3;
% 0 -0.13 0.68;
% 0.46 0.81 -0.02];
% Radii = [0.2; 0.5; 0.8; 0.4];
Centers = round(N * [.35 .25 .35; .5 .5 .5; .35 .75 .35]);
Radii   = N * [.1875; .25; .1875];

%n=linspace(-1,1,100); 

n=1:N;
[x,y,z]=ndgrid(n,n,n);

Cube = zeros(size(x)); % Pre-allocating SFEAR

for i=1:size(Centers,2)
  sfear = (x-Centers(i,1)).^2 + (y-Centers(i,2)).^2 +(z-Centers(i,3)).^2 <= Radii(i)^2;
  Cube=Cube + sfear;
end
save(['bamba' int2str(N)], 'Cube');
isosurface(sfear,0)