msg=$1  # $1为第一个参数
if  test $msg -eq 
then
	echo "执行git pull"
    git pull
else
	echo "执行git add ."
    git add .
	echo "执行git commit -m ${msg}"
    git commit -m"${msg}"
	echo "执行git push"
    git push
fi