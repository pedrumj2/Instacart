#!/bin/bash



if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi
user=$(logname)
sudo -u $user  git submodule update --init --recursive
(cd LinuxVMSetup
  sudo -u $user git checkout master
  sudo -u $user git pull  
)
(cd mysql_csv_import
  sudo -u $user git checkout master
  sudo -u $user git pull
)
#(cd LinuxVMSetup
#  chmod +x deep_learning_setup.sh
#  ./deep_learning_setup.sh
#)

(cd SetupScripts
  #./setupEnv.sh
  ./minimizeDataSet.sh 2
)

