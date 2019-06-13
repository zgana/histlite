# CMakeLists.txt
# histlite

i3_project (histlite
    PYTHON_DEST icecube/histlite
    PYTHON_DIR histlite
    DOCS_DIR resources/docs
)

i3_test_scripts (
    resources/test/test_Hist.py
)

# vim: nospell tw=0
