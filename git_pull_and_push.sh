msg=$1  # $1为第一个参数
if  [ $msg = pull ];then
	echo "执行git pull"
    git pull

elif [ $msg = update ];then
	echo "执行git add ."
    git add .
	echo "执行git commit -m ${msg}"
    git commit -m "${msg}"
	echo "执行git push"
    git push
else
	echo "执行hexo generate"
    hexo g
        echo "执行hexo deploy"
    hexo d
	echo "执行git add ."
    git add .
	echo "执行git commit -m ${msg}"
    git commit -m"${msg}"
	echo "执行git push"
    git push git@github.com:messj-0508/messj-0508.github.io.git hexo
fi
