# These commands maximize exposure at the device's max fps (= 30)

v4l2-ctl --device=1 \
	--set-ctrl=exposure_auto=1 \
	--set-ctrl=white_balance_temperature_auto=0
v4l2-ctl --device=1 \
	--set-ctrl=exposure_absolute=78
