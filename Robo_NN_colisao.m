%---------------------------------------------
%Cinemática inversa robo baseado em modelo NN
%---------------------------------------------

%Distancia dos bracos do robo
%Precisam ser >= maior distancia do espaco
a=3.55;
b=3.55;
braco=[0;a;a+b];
origem=[0 0];

%Dimensão do espaço
dx=11; %vai de -5 a 5 (a origem tambem conta)
dy=11; %vai de -5 a 5

%Visual (Iniciar figura)
x=0;
y=0;
figure(1);
h=plot(braco,[0;0;0],'k','LineWidth',3);
axis([-10 10 -10 10]);
grid on;
hold on;

%obstaculos
rectangle('Position',[1 1 1 1],'FaceColor','b');
%Plots para interseção
xi=[1 2];
xi2=[2 1];
plot(xi);
plot(xi2);

%--------------------------------
%Posicionamento dos braços robo
%--------------------------------

x=-5; %De modo a comecar no canto sup esquerdo
y=6;  %De modo a comecar no canto sup esquerdo
coli=1; %incremento matriz colisoes
nrs=0.001; %tender para 0
Mt=[]; %Matriz guarda x,y e coordenadas inversas
Mts=[]; % Matriz info das coordenadas com simetria braços
for i=1:dy
    y=y-1;
    x=-5;
    for j=1:dx
        % Cinematica Inversa
        aux=(x.^2+y.^2-a*a-b*b)/(2*a*b);
        if abs(aux)<=1
         beta=acos(aux);
         betas=-acos(aux);
        end
        beta=acos(aux);
        betas=-acos(aux);
        alfa=atan2(y,x)-atan2(b*sin(beta),a+b*cos(beta));
        alfas=atan2(y,x)-atan2(b*sin(betas),a+b*cos(betas));
        
        %Cinematica directa
        % Coordenadas do ponto Q (Articulacao medial)
        Qx=a*cos(alfa);
        Qy=a*sin(alfa);
        
        Qxs=a*cos(alfas);
        Qys=a*sin(alfas);
        
        % Coordenadas do ponto P
        Px=round(Qx+b*cos(alfa+beta));
        Py=round(Qy+b*sin(alfa+beta));
        
        Pxs=round(Qxs+b*cos(alfas+betas));
        Pys=round(Qys+b*sin(alfas+betas));
           
        set(h(1),'XData',[0;Qx;Px],'YData',[0;Qy;Py]);
        plot(Px,Py,'r.');
        x=x+1;
        
        Mt=[Mt;Px,Py,alfa,beta];
        Mts=[Mts;Pxs,Pys,alfas,betas];
        
        %Colisao
        sr_ia=[1 1]; %segmento recta intersecao
        sr_ib=[2 2];
        sr_ic=[2 1];
        sr_id=[1 2];
        Q=[Qx,Qy];
        Qs=[Qxs,Qys];
        P=[Px,Py];
        Ps=[Pxs,Pys];
        Mcalc=[]; %Matriz segmentos
        
        for iii=1:4
            
            if iii==3
                Q=Qs;
                P=Ps;
                sr_ic=sr_ia;
                sr_id=sr_ib;
            end
            
            [dist,alfaa,betaa]=dist2segm(origem',Q',sr_ia',sr_ib');
            Mcalc=[Mcalc;dist];
            [dist,alfaa,betaa]=dist2segm(Q',P',sr_ia',sr_ib');
            Mcalc=[Mcalc;dist];
            sr_ia=sr_ic;
            sr_ib=sr_id;
        end
        
        if Mcalc(1,1)<nrs || Mcalc(3,1)<nrs || Mcalc(2,1)<nrs|| Mcalc(4,1)<nrs ...
                Px==1 && Py==1 || Px==1 && Py==2 || Px==2 && Py==1 || Px==2 && Py==2;
            
            Mt(coli,1:4)=[Px,Py,Inf,Inf];
            plot(Px,Py,'gx');
        end
        if Mcalc(5,1)<nrs || Mcalc(7,1)<nrs || Mcalc(6,1)<nrs || Mcalc(8,1)<nrs ...
                Pxs==1 && Pys==1 || Pxs==1 && Pys==2 || Pxs==2 && Pys==1 || Pxs==2 && Pys==2;
            
            Mts(coli,1:4)=[Pxs,Pys,Inf,Inf];
            plot(Px,Py,'bx');
        end
        coli=coli+1; %incremento actualiz matriz
    end
end

%--------------------------------
%Redes Neurais
%--------------------------------

Mnn=[]; %Matriz das aprendizagens
sMt=size(Mt,1); %nr linhas da matriz Mt
for inn=1:sMt

x=Mt(inn,1:2); %entrada NN
y=Mt(inn,3:4); %saida NN

np=size(x,2);

net = newff(x,y,4);
net = train(net,x,y);
yN = net(x); %output da rede neural
errors = yN - y; %erro entre saida rede neural e o que era suposto dar
E2=sum(errors.^2)/np; %Erro quadrático

if x(1,1)==x(1,2)
    yN=yN-errors; %Corrigir erro quando linha e coluna da entrada igual
end

Mnn=[Mnn;Mt(inn,1),Mt(inn,2),yN(1,1),yN(1,2)]; %Matriz aprendizagem NN
end

Mnns=[];
sMts=size(Mts,1); %nr linhas da matriz Mt
for inn=1:sMts

x=Mts(inn,1:2); %entrada NN
y=Mts(inn,3:4); %saida NN

np=size(x,2);

net = newff(x,y,4);
net = train(net,x,y);
yN = net(x); %output da rede neural
errors = yN - y; %erro entre saida rede neural e o que era suposto dar
E2=sum(errors.^2)/np; %Erro quadrático

if x(1,1)==x(1,2)
    yN=yN-errors; %Corrigir erro quando linha e coluna da entrada igual
end

Mnns=[Mnns;Mts(inn,1),Mts(inn,2),yN(1,1),yN(1,2)]; %Matriz aprendizagem NN
end

%-------------------------------
%Distancia entre dois segmentos
%-------------------------------

function [dist,alfaa,betaa]=dist2segm(A,B,C,D)
% Calcula a distância entre o segmento AB com o segmento CD

AB=B-A;
CD=D-C;
CA=A-C;
CD=D-C;
AD=D-A;

m2AB=AB'*AB;
m2CD=CD'*CD;

den=m2AB*m2CD-(AB'*CD).^2;

cos_ang2segm= 1-AB'*CD/sqrt(m2AB*m2CD);
if abs(cos_ang2segm)<1e-3
    % Rectas paralelas:
    % Calcular a equação da recta CD e depois calcular a distância do ponto
    % A a essa recta
    % recta s: (yD-yC)*x-(xD-xC)*y-(yD-yC)*xC-(xD-xC)*yC=0
    %              a x + b y + c =0
    [dist1,alfa1]=dist_point2segm(C,A,B);
    [dist2,alfa2]=dist_point2segm(D,A,B);
    if dist1<dist2
        dist=dist1;
        alfaa=alfa1;
    else
        dist=dist2;
        alfaa=alfa2;
    end
    
    [dist1,beta1]=dist_point2segm(A,C,D);
    [dist2,beta2]=dist_point2segm(B,C,D);
    if dist1<dist2
        dist=min(dist,dist1);
        betaa=beta1;
    else
        dist=min(dist,dist2);
        betaa=beta2;
    end
    
else
    alfaa=(-m2CD*(AB'*CA)+(AB'*CD)*(CD'*CA))/den;
    betaa=(m2AB*(CD'*CA)-(CD'*AB)*(AB'*CA))/den;
    
    % Caso 1:
    if ~(alfaa>=0 && betaa >=0 && alfaa<=1 && betaa<=1)
        if alfaa>1
            a=(alfaa-1)*sqrt(m2AB);
        else
            if alfaa<0
                a=alfaa*sqrt(m2AB);
            else
                a=0; %min(abs([alfa,alfa-1]))*sqrt(m2AB);
            end
        end
        if betaa>1
            b=(betaa-1)*sqrt(m2CD);
        else
            if betaa<0
                b=betaa*sqrt(m2CD);
            else
                b=0; %min(abs([beta,beta-1]))*sqrt(m2CD);
            end
        end
        
        % Caso 2A
        if abs(b)>abs(a)
            if betaa>1
                betaa=1;
            else
                if betaa<0
                    betaa=0;
                end
            end
            alfaa=(betaa*AB'*CD-AB'*CA)/m2AB;
            if alfaa<0
                alfaa=0;
            else
                if alfaa>1
                    alfaa=1;
                end
            end
            betaa=(CD'*CA+alfaa*CD'*AB)/m2CD;
            if betaa>1
                betaa=1;
            else
                betaa=0;
            end
        else
            % Caso 2B
            if alfaa>1
                alfaa=1;
            else
                if alfaa<0
                    alfaa=0;
                end
            end
            betaa=(CD'*CA+alfaa*CD'*AB)/m2CD;
            if betaa<0
                betaa=0;
            else
                if betaa>1
                    betaa=1;
                end
            end
            alfaa=(betaa*AB'*CD-AB'*CA)/m2AB;
            if alfaa>1
                alfaa=1;
            else
                if alfaa<0
                    alfaa=0;
                end
            end
        end
    end
    P=A+alfaa*AB;
    Q=C+betaa*CD;
    
    dist=sqrt((Q-P)'*(Q-P));
end
end

function [D,alfaa]=dist_point2segm(P,A,B)
% Calcula a distância de um ponto P ao segmento AB

PA=A-P;
AB=B-A;
alfaa=-AB'*PA/(AB'*AB);

if alfaa<0
    alfaa=0;
else
    if alfaa>1
        alfaa=1;
    end
end
Q=A+alfaa*AB;
D=sqrt((Q-P)'*(Q-P));
end