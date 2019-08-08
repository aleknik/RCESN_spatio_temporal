function [dx, dy,dz] = step(x_vec,y_mat,z_mat,K,J,I)
xvec=repmat(x_vec,1,3);
ymat=repmat(y_mat,3,1);
zmat=repmat(z_mat,1,1,3);
F=20; c=10; b=10; h=1;
e=10; d=10;

x_minus=xvec(K:(2*K-1));
x_minus2=xvec((K-1):(2*K-2));
x_plus=xvec((K+2):(2*K+1));

y_minus=ymat(J:(2*J-1),:);
y_plus=ymat((J+2):(2*J+1),:);
y_plus2=ymat((J+3):(2*J+2),:);

z_minus=zmat(:,:,I:(2*I-1));
z_minus2=zmat(:,:,(I-1):(2*I-2));
z_plus=zmat(:,:,(I+2):(2*I+1));

y_k=sum(y_mat);

z_kj=sum(z_mat,3);

dx=x_minus.*(x_plus-x_minus2)-x_vec...
    +F-(h*c/b)*y_k;
dy=-c*b*y_plus.*(y_plus2-y_minus)...
    -c*y_mat+(h*c/b)*x_vec-(h*e/d)*z_kj;

dz=e*d*z_minus.*(z_plus-z_minus2)...
    -e*z_mat+(h*e/d)*y_mat;

end