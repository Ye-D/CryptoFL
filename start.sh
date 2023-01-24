#!/bin/sh
for i in {0..8}; do
    python xx_client.py --id $i &
done &
python xx_client.py --id 9