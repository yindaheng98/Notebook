K=1:1:20;
N=100000;
l={};
for M=100000:100000:1000000
    P=(1-(1-1./M).^(K.*N)).^K;
    plot(K,P);
    l{end+1}=sprintf('M=%.0e',M);
    hold on
end
legend(l)
xlabel('hash函数数量K')
ylabel('假正率P')