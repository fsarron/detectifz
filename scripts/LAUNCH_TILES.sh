#!/bin/bash

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# use here your expected variables
echo "WORKDIR = $WORKDIR"
echo "SCRIPTS_DIR = $SCRIPTS_DIR"
echo "EXTRA_VALUES = $EXTRA_VALUES"

#detectifz_dir = ${1}
#detectifz_exec = ${2}


##TO DO PRIOR TO LAUNCH
#mkdir $WORKDIR
#cp $SCRIPTS_DIR/config_tile.py $WORKDIR
#cp $SCRIPTS_DIR/config_master_detectifz_GAEA.py $WORKDIR/config_master_detectifz.py

cd $WORKDIR
#cp $SCRIPTS_DIR/run_tiling.py .

#field=`cat config_tile.py | grep 'field =' | awk '{print $3}'`
#field="${field#?}" # removes first character
#field="${field%?}"  # removes last character
field='H15_UDS'
echo
#cp $SCRIPTS_DIR/config_master_detectifz_GAEA.py $WORKDIR/${field}_DETECTIFzrun/config_master_detectifz.py

#python run_tiling.py -c config_tile.py

for i in $(ls -d ${field}_DETECTIFzrun/*/)
do 
    tileid=`echo ${i%%/} | awk -F / '{print $2}'`
    echo $tileid
    cd $WORKDIR/${i}
    echo pwd
    cp $SCRIPTS_DIR/run_detectifz_tile.py .
    echo python run_detectifz_tile.py -c config_detectifz.py -t $tileid
    SECONDS=0
    python run_detectifz_tile.py -c config_detectifz.py -t $tileid
    DURATION_IN_SECONDS=` echo $tileid completed in $SECONDS seconds`
    echo
    echo $DURATION_IN_SECONDS
    echo
done





###################################



#i=19
#LISTTILES=`ls ${field}_DETECTIFzrun/tile* | grep tile`
#echo $LISTTILES
#lastTile=${LISTTILES: -5:4}
#echo $lastTile

#for i in {0000..$lastTile}
#for (( c=0; c<=$lastTile+2; c++ ))
#do
#   echo "Welcome $c times"
#done


#echo $lastFour

#tiles=`echo $LISTTILES | awk -F : '{print $0}'`;





#tiles=`echo $LISTTILES | awk -F : '{print $'$i'}'`;

#files=`ls -l ${field}_DETECTIFzrun/tile* | awk '{print $1}'`
#echo $tiles


#for filter in $FILTERS_IRAC; do
#    echo ""
#    echo "convolve KERNEL IRAC"
#    echo $filter
#    python $python_exec $rootdir $cluster $filter $tag $psftargetsize_IRAC
#done