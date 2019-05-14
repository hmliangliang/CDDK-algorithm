data=shuttle;%data set is shuttle
k=4;
col=size(data,2);
winsize=2500;%滑动窗口的大小
a=0.01;%a代表显著性水平
n=20000;
klabel=length(unique(data(:,col)));%不同类标签的个数
ensemble=struct('traindata',[],'traintarget',[],'weight',0);
acc=[];%保存每一个数据块的子分类器中分类准确率的最大值
cc=0;%记录测试的次数
accuracy=[];%保存数据测试结果的准确率
tic;
for i=1:size(data,1)
    if mod(i,2*winsize)==0
        %获取最大的准确率的平均值
        if isempty(acc)==1
            pm=0;
        else
            pm=mean(acc);
        end
        pc=0;%保存临时子分类器的最大值
        %获取训练集和测试集
        traindata=data((i-2*winsize+1):(i-winsize),1:(col-1));
        traintarget=data((i-2*winsize+1):(i-winsize),col);
        testdata=data((i-winsize+1):i,1:(col-1));
        testtarget=data((i-winsize+1):i,col);
        %创建临时分类器
        temp=struct('traindata',[],'traintarget',[],'weight',0);
        temp.traindata=traindata;
        temp.traintarget=traintarget;
        if (size(ensemble,2)==1)&&(isempty(ensemble(1).traindata)==1)%说明当前分类器系统为空,先添加一个分类器
            ensemble(1).traindata=traindata;
            ensemble(1).traintarget=traintarget;
            ensemble(1).weight=1;
        else %先训练后测试
            result=[];%保存每个子分类器的分类结果
            weight=zeros(size(traindata,1),klabel);%保存记录的投票结果权值累积
            for x=1:size(ensemble,2) %对于每个分类器而言
                model=ClassificationTree.fit(ensemble(x).traindata,ensemble(x).traintarget);
                res=predict(model,traindata);
                result=[result,res];%行代表事例，列代表分类器
            end
            %统计投票结果
            for wi=1:size(result,1)
                for wj=1:size(result,2)
                    weight(wi,result(wi,wj))= weight(wi,result(wi,wj))+ensemble(wj).weight;
                end
            end
            [c,label]=max(weight,[],2);%c代表权值，label记录最终的投票结果
            count=0;
            for q=1:size(label,1)%计算加权投票分类结果的错误率
                if label(q,1)==traintarget(q,1)
                    count=count+1;
                end
            end
            p=count/(size(traindata,1));%p为集成式分类器对当前数据块分类结果的准确率
            acc=[acc,p];
            ka=(pm-p)/(1-p);%计算集成式分类器的Kappa系数
            %更新分类器的权值
            for ej=1:size(result,2)
                ecount=0;
                for ei=1:size(result,1)%计算当前子分类器分类的准确率
                    if result(ei,ej)==traintarget(ei,1)
                        ecount=ecount+1;
                    end
                end
                epi=ecount/(size(result,1));%子分类器当前分类的准确率
                ki=(pm-epi)/(1-pm);%计算子分类器分类结果的Kappa系数
                ensemble(ej).weight=log2((1+ki)/ki);
            end
            %检测概念漂移
            theta=(1/(1-p))*sqrt(((-3)*log(a))/n);
            if ka>theta %发生概念漂移
                %删除所有Kappa系数<theta的子分类器
                disp(['在',num2str(i),'处发生了概念漂移']);
                t=1;
                while t<size(ensemble,2)
                    modell=ClassificationTree.fit(ensemble(t).traindata,ensemble(t).traintarget);
                    re=predict(modell,traindata);
                    %统计正确率
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
                    %删除权值最小的分类器
                      [c,d]=min([ensemble.weight]);
                      ensemble(d)=[];
                      temp.weight=1;
                      ensemble=[ensemble,temp];
                end
                acc=[];%将保存的历史准确率清空
            else %未发生概念漂移
                if size(ensemble,2)<k
                    if isempty(ensemble(1).traindata)==1
                        ensemble(1)=temp;
                        ensemble(1).weight=log(1/(pm+1));
                    else
                      %删除权值最小的分类器
                      temp.weight=1;
                      ensemble=[ensemble,temp];
                    end
                else
                     %删除权值最小的分类器
                      [c,d]=min([ensemble.weight]);
                      ensemble(d)=[];
                      temp.weight=1;
                      ensemble=[ensemble,temp];
                end
            end
            for as=1:size(ensemble,2)%用当前数据块去训练每一个分类器
                ensemble(as).traindata=[ensemble(as).traindata;traindata];
                ensemble(as).traintarget=[ensemble(as).traintarget;traintarget];
            end
        end
        % 利用训练的模型进行测试
        cc=cc+1;
        tcount=0;
        cresult=[];%记录集成式分类器每个子分类器对数据块的分类结果
        cweight=zeros(size(testdata,1),klabel);
        for ci=1:size(ensemble,2)
            cmodel=ClassificationTree.fit(ensemble(ci).traindata,ensemble(ci).traintarget);
            cres=predict(cmodel,testdata);
            cresult=[cresult,cres];
        end
        %计算累计投票权值
        for x=1:size(cresult,1)
            for y=1:size(cresult,2)
                cweight(x,cresult(x,y))=cweight(x,cresult(x,y))+ensemble(y).weight;
            end
        end
        %作出决策
        [c,clast]=max(cweight,[],2);
        for z=1:size(clast,1)%计算正确率
            if clast(z,1)==testtarget(z,1)
                tcount=tcount+1;
            end
        end
        right=tcount/(size(testdata,1));
        accuracy=[accuracy,right];
        disp(['第',num2str(cc),'次测试的准确率为:',num2str(right)]);
    end
end
lastright=mean(accuracy);
disp(['CDDK算法在当前数据集上测试结果的准确率为：',num2str(lastright)]);
toc;

