module Device

    using CUDA

    export to_device, isgpu

    # Return x on GPU if CUDA is available
    to_device(x) = CUDA.functional() ? cu(x) : x
    isgpu() = CUDA.functional()

end
