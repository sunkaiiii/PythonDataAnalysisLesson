#coding=utf-8

num=[1,2,3,4]
def func1():
    count=0
    for i in range(0,4):
        for j in range(0,4):
            if j==i:
                continue
            for k in range(0,4):
                if k==j or k==i:
                    continue
                print(num[i]*100+num[j]*10+num[k])
                count=count+1
    print("一共有数字"+str(count)+"个")

def func2(num1,num2):
    if num1==0 or num2==0:
        print("最小公约数是" + str(0))
        return
    while(1):
        if num1%num2==0:
            break
        a=num1%num2
        num1=num2
        num2=a
        if a==0:
            break
    print("最小公约数是"+str(num1 if num1==0 else num2))

func1()
func2(0,0)