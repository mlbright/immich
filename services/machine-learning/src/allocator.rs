pub fn release() {
    unsafe {
        libmimalloc_sys::mi_collect(true);
    }
}
