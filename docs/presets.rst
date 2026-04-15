Foundation Model Presets
=======================

SPEED includes ready-to-use preprocessing configs for popular EEG foundation models
in ``configs/presets/``.

.. list-table::
   :header-rows: 1
   :widths: 15 15 10 10 10 40

   * - Preset
     - Bandpass
     - Resample
     - ICA
     - Windows
     - Target Model
   * - ``labram.yaml``
     - 0.1--75 Hz
     - 200 Hz
     - Off
     - 60s
     - LaBraM
   * - ``cbramod.yaml``
     - 0.5--50 Hz
     - 200 Hz
     - Off
     - 30s
     - CBraMod
   * - ``biot.yaml``
     - 0.5--45 Hz
     - 200 Hz
     - Off
     - 10s
     - BIOT
   * - ``reve.yaml``
     - 0.5--100 Hz
     - 256 Hz
     - Off
     - 60s
     - REVE

Usage
-----

.. code-block:: bash

   python scripts/preprocess.py --config configs/presets/labram.yaml \
       --dataset_path /data/tuh/ --out_path /data/tuh_labram/
