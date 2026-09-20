// Offline schedule probe. Acceptance counts are diagnostic inputs, never model data.
#include "mtp_draft_schedule.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) try {
    if (argc != 2) return 2;
    qrt_mtp_draft_schedule::Schedule schedule;
    if (!schedule.reset(std::stoull(argv[1]))) return 2;
    unsigned int accepted = 0, batches = 0;
    while (std::cin >> accepted) {
        if (++batches > 1024u) return 2;
        qrt_mtp_draft_schedule::Batch batch;
        if (!schedule.begin(&batch)) return 2;
        std::cout << "{\"first_position\":" << batch.first_position
                  << ",\"scheduled_rows\":" << batch.scheduled_rows
                  << ",\"speculative\":" << (batch.speculative ? "true" : "false")
                  << ",\"accepted_rows\":" << accepted << "}\n";
        if (!accepted) return 0; // Last observed batch has no subsequent position.
        if (!schedule.complete(accepted)) return 2;
    }
    return std::cin.eof() && batches ? 0 : 2;
} catch (...) { return 2; }
