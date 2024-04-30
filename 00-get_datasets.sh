if [ ! -d "Dataset_1_NCA_battery" ]; then
  wget https://zenodo.org/records/6405084/files/Dataset_1_NCA_battery.zip -O Dataset_1_NCA_battery.zip
  tar -xzvf Dataset_1_NCA_battery.zip
  rm Dataset_1_NCA_battery.zip
fi
