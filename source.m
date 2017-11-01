function s = source(t, xs, xd, D, t0, u)
s = (xs(1)./(4*pi*sqrt(D(1)*D(2))*(t-t0))).*exp(-(((xd(1)-(xs(2) + u*(t-t0))).^2)./(4*D(1)*(t-t0)))).*exp(-((xd(2)- xs(3)).^2)./(4*D(2)*(t-t0)));
end