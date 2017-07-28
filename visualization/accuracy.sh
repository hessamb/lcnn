SCRIPTPATH=$(cd $(dirname $0) && pwd -P)
logfile=$1
if [ -d $logfile ]; then
   logfile=$logfile/log.txt
fi
grep -ae "TRAINING SUMMARY\|TESTING SUMMARY" $logfile | awk '{if(/TRAINING/) print $NF; else print $15}' | sed '$!N;s/\n/ /' | python $SCRIPTPATH/plot.py 'train' 'test' &
