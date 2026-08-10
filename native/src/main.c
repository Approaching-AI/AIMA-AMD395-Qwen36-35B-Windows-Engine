#include "qrt.h"

#include <stdio.h>

int main(void) {
    const qrt_target_contract_t *contract = qrt_target_contract();
    if (contract == NULL) {
        fputs("contract=null\n", stderr);
        return 1;
    }

    printf("repo=%s\n", contract->repo);
    printf("model=%s\n", contract->model);
    printf("precision=%s\n", contract->precision);
    printf("runtime_language=%s\n", contract->primary_runtime_language);
    printf("tooling_language=%s\n", contract->tooling_language);
    printf("target_os=%s\n", contract->target_os);
    printf("target_device=%s\n", contract->target_device);
    printf("test_host=%s\n", contract->test_host);
    printf("dependency_policy=%s\n", contract->dependency_policy);
    printf("status=%s\n", contract->status);
    return 0;
}
