data=shuttle;%data set is shuttle
k=4;
col=size(data,2);
winsize=2500;%�������ڵĴ�С
a=0.01;%a����������ˮƽ
n=20000;
klabel=length(unique(data(:,col)));%��ͬ���ǩ�ĸ���
ensemble=struct('traindata',[],'traintarget',[],'weight',0);
acc=[];%����ÿһ�����ݿ���ӷ������з���׼ȷ�ʵ����ֵ
cc=0;%��¼���ԵĴ���
accuracy=[];%�������ݲ��Խ����׼ȷ��
tic;
for i=1:size(data,1)
    if mod(i,2*winsize)==0
        %��ȡ����׼ȷ�ʵ�ƽ��ֵ
        if isempty(acc)==1
            pm=0;
        else
            pm=mean(acc);
        end
        pc=0;%������ʱ�ӷ����������ֵ
        %��ȡѵ�����Ͳ��Լ�
        traindata=data((i-2*winsize+1):(i-winsize),1:(col-1));
        traintarget=data((i-2*winsize+1):(i-winsize),col);
        testdata=data((i-winsize+1):i,1:(col-1));
        testtarget=data((i-winsize+1):i,col);
        %������ʱ������
        temp=struct('traindata',[],'traintarget',[],'weight',0);
        temp.traindata=traindata;
        temp.traintarget=traintarget;
        if (size(ensemble,2)==1)&&(isempty(ensemble(1).traindata)==1)%˵����ǰ������ϵͳΪ��,�����һ��������
            ensemble(1).traindata=traindata;
            ensemble(1).traintarget=traintarget;
            ensemble(1).weight=1;
        else %��ѵ�������
            result=[];%����ÿ���ӷ������ķ�����
            weight=zeros(size(traindata,1),klabel);%�����¼��ͶƱ���Ȩֵ�ۻ�
            for x=1:size(ensemble,2) %����ÿ������������
                model=ClassificationTree.fit(ensemble(x).traindata,ensemble(x).traintarget);
                res=predict(model,traindata);
                result=[result,res];%�д����������д��������
            end
            %ͳ��ͶƱ���
            for wi=1:size(result,1)
                for wj=1:size(result,2)
                    weight(wi,result(wi,wj))= weight(wi,result(wi,wj))+ensemble(wj).weight;
                end
            end
            [c,label]=max(weight,[],2);%c����Ȩֵ��label��¼���յ�ͶƱ���
            count=0;
            for q=1:size(label,1)%�����ȨͶƱ�������Ĵ�����
                if label(q,1)==traintarget(q,1)
                    count=count+1;
                end
            end
            p=count/(size(traindata,1));%pΪ����ʽ�������Ե�ǰ���ݿ��������׼ȷ��
            acc=[acc,p];
            ka=(pm-p)/(1-p);%���㼯��ʽ��������Kappaϵ��
            %���·�������Ȩֵ
            for ej=1:size(result,2)
                ecount=0;
                for ei=1:size(result,1)%���㵱ǰ�ӷ����������׼ȷ��
                    if result(ei,ej)==traintarget(ei,1)
                        ecount=ecount+1;
                    end
                end
                epi=ecount/(size(result,1));%�ӷ�������ǰ�����׼ȷ��
                ki=(pm-epi)/(1-pm);%�����ӷ�������������Kappaϵ��
                ensemble(ej).weight=log2((1+ki)/ki);
            end
            %������Ư��
            theta=(1/(1-p))*sqrt(((-3)*log(a))/n);
            if ka>theta %��������Ư��
                %ɾ������Kappaϵ��<theta���ӷ�����
                disp(['��',num2str(i),'�������˸���Ư��']);
                t=1;
                while t<size(ensemble,2)
                    modell=ClassificationTree.fit(ensemble(t).traindata,ensemble(t).traintarget);
                    re=predict(modell,traindata);
                    %ͳ����ȷ��
                    pcount=0;
                    for si=1:size(traindata,1)
                        if re(si,1)==traintarget(si,1)
                            pcount=pcount+1;
                        end
                    end
                    pright=pcount/(size(traindata,1));
                    pk=(pm-pright)/(1-pright);
                    if pk>(1/(1-pk))*sqrt(((-3)*log(a))/(pm*(n^3)))
                        ensemble(t)=[];
                    else
                        t=t+1;
                    end
                end
                if size(ensemble,2)<k
                    if size(ensemble,2)==0
                        ensemble(1)=temp;
                        ensemble(1).weight=1;
                    else
                      temp.weight=log(1/(pm+1));
                      ensemble=[ensemble,temp];
                    end
                else
                    %ɾ��Ȩֵ��С�ķ�����
                      [c,d]=min([ensemble.weight]);
                      ensemble(d)=[];
                      temp.weight=1;
                      ensemble=[ensemble,temp];
                end
                acc=[];%���������ʷ׼ȷ�����
            else %δ��������Ư��
                if size(ensemble,2)<k
                    if isempty(ensemble(1).traindata)==1
                        ensemble(1)=temp;
                        ensemble(1).weight=log(1/(pm+1));
                    else
                      %ɾ��Ȩֵ��С�ķ�����
                      temp.weight=1;
                      ensemble=[ensemble,temp];
                    end
                else
                     %ɾ��Ȩֵ��С�ķ�����
                      [c,d]=min([ensemble.weight]);
                      ensemble(d)=[];
                      temp.weight=1;
                      ensemble=[ensemble,temp];
                end
            end
            for as=1:size(ensemble,2)%�õ�ǰ���ݿ�ȥѵ��ÿһ��������
                ensemble(as).traindata=[ensemble(as).traindata;traindata];
                ensemble(as).traintarget=[ensemble(as).traintarget;traintarget];
            end
        end
        % ����ѵ����ģ�ͽ��в���
        cc=cc+1;
        tcount=0;
        cresult=[];%��¼����ʽ������ÿ���ӷ����������ݿ�ķ�����
        cweight=zeros(size(testdata,1),klabel);
        for ci=1:size(ensemble,2)
            cmodel=ClassificationTree.fit(ensemble(ci).traindata,ensemble(ci).traintarget);
            cres=predict(cmodel,testdata);
            cresult=[cresult,cres];
        end
        %�����ۼ�ͶƱȨֵ
        for x=1:size(cresult,1)
            for y=1:size(cresult,2)
                cweight(x,cresult(x,y))=cweight(x,cresult(x,y))+ensemble(y).weight;
            end
        end
        %��������
        [c,clast]=max(cweight,[],2);
        for z=1:size(clast,1)%������ȷ��
            if clast(z,1)==testtarget(z,1)
                tcount=tcount+1;
            end
        end
        right=tcount/(size(testdata,1));
        accuracy=[accuracy,right];
        disp(['��',num2str(cc),'�β��Ե�׼ȷ��Ϊ:',num2str(right)]);
    end
end
lastright=mean(accuracy);
disp(['CDDK�㷨�ڵ�ǰ���ݼ��ϲ��Խ����׼ȷ��Ϊ��',num2str(lastright)]);
toc;

