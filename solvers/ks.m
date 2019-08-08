N = 64;  d = 22;  h = 0.25;  nstp = 1000;  np = 1;
a0 = zeros(N-2,1);  a0(1:6) = 0.6;
[tt, aa] = ksfmstp(a0, d, h, nstp, np);
[xx, uu] = ksfm2real(aa, d);

csvwrite('ks_test.csv', uu)