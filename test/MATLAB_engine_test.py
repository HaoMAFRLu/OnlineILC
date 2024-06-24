import matlab.engine

def test_matlab_engine():
    try:
        eng = matlab.engine.start_matlab()
        
        result = eng.sqrt(4.0)
        
        print(f"The square root of 4.0 is {result}")
        
        eng.quit()
        
        print("MATLAB Engine API is installed and working correctly.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_matlab_engine()
