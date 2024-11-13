Usage
=====

This section provides instructions on how to use the NDVI processing pipeline.

Fetching NDVI Data
------------------

To fetch NDVI data for a specific date, run:

.. code-block:: bash

    python scripts/fetch_ndvi.py YYYY-MM-DD

Processing NDVI Data
--------------------

To process the fetched NDVI data with cloud masking, run:

.. code-block:: bash

    python scripts/process_ndvi.py YYYY-MM-DD

Comparing NDVI Data
-------------------

To compare NDVI data across multiple years, run:

.. code-block:: bash

    python scripts/compare_ndvi.py YEAR1 YEAR2 YEAR3 YEAR4 YEAR5

Visualizing NDVI Trends
-----------------------

To launch the Streamlit application for visualization, run:

.. code-block:: bash

    streamlit run streamlit_app/visualization_app.py
