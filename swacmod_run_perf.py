import swacmod.feature_flags as ff

def run_main():
    ff.turn_perf_features_on()
    import swacmod_run
    swacmod_run.run_main()

if __name__ == "__main__":
    run_main()