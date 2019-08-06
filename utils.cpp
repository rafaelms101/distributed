#include "utils.h"

#include <sys/time.h>
#include <cstdlib>
#include <cassert>
#include <sys/stat.h>
#include <cstring>
#include <fstream>

#include "config.h"
#include "readSplittedIndex.h"

double now() {
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}


// not very clean, but works as long as sizeof(int) == sizeof(float)
int *ivecs_read(const char *fname, int *d_out, int *n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

float * fvecs_read (const char *fname, int *d_out, int *n_out) {
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;	

    assert(fread(&d, 1, sizeof(int), f) == sizeof(int) || !"could not read d");
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = (int) d; *n_out = (int) n;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for(size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

std::vector<double> load_prof_times(Config& cfg) {
	char file_path[100];
	sprintf(file_path, "prof/%d_%d_%d_%d_%d_%d", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size);
	std::ifstream file;
	file.open(file_path);

	if (! file.good()) {
		std::printf("File prof/%d_%d_%d_%d_%d_%d not found", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size);
		std::exit(- 1);
	}

	int total_size;
	file >> total_size;

	std::vector<double> times(total_size + 1);

	times[0] = 0;

	for (int i = 1; i <= total_size; i++) {
		file >> times[i];
	}

	file.close();
	
	return times;
}

faiss::IndexIVFPQ* load_index(float start_percent, float end_percent, Config& cfg) {
	deb("Started loading");

	char index_path[500];
	sprintf(index_path, "%s/index_%d_%d_%d", INDEX_ROOT, cfg.nb, cfg.ncentroids, cfg.m);

	deb("Loading file: %s", index_path);

	FILE* index_file = fopen(index_path, "r");
	auto cpu_index = static_cast<faiss::IndexIVFPQ*>(read_index(index_file, start_percent, end_percent));

	deb("Ended loading");

	//TODO: Maybe we shouldnt set nprobe here
	cpu_index->nprobe = cfg.nprobe;
	return cpu_index;
}
