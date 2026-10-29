// Decay GENIE's undecayed taus with TAUOLA++ (tau gun mode).
//
//   tau_decay <taus.txt> <daughters.txt> [seed]
//
// Input: one tau per line, "iev pdg px py pz E helicity" (GeV; lines starting
// with '#' are skipped).  helicity is GENIE's polarization vector projected on
// the tau direction, -1 for a tau- and +1 for a tau+ in GENIE's massless-lepton
// approximation.
//
// Tauola::decayOne takes the polarization in the tau rest frame with z along
// the lab tau direction, but with the pair convention of the interface manual
// (appendix "tau gun"): polz = +1 is a LEFT-handed tau- and a RIGHT-handed
// tau+.  So polz = -helicity for a tau- and +helicity for a tau+.  Checked:
// tau- -> pi- nu with polz = +1 gives <E_pi/E_tau> = 1/3, the soft pion a
// left-handed tau- must give; polz = -1 gives 2/3.
//
// Output: one stable daughter per line, "iev pdg px py pz E" (GeV), neutrinos
// included, in the lab frame.  Photons from TAUOLA's radiative corrections are
// kept.  A summary of the decay modes goes to stderr.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "Tauola/Tauola.h"
#include "Tauola/TauolaHEPEVTEvent.h"
#include "Tauola/TauolaHEPEVTParticle.h"

using namespace Tauolapp;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <taus.txt> <daughters.txt> [seed]\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  std::ofstream out(argv[2]);
  if (!in || !out) {
    std::cerr << "cannot open " << (in ? argv[2] : argv[1]) << "\n";
    return 1;
  }
  const int seed = argc > 3 ? std::atoi(argv[3]) : 4357;

  // All decay channels, matrix-element weighted; radiative corrections on.
  Tauola::setDecayingParticle(15);
  Tauola::setSameParticleDecayMode(0);
  Tauola::setOppositeParticleDecayMode(0);
  Tauola::setSeed(seed, 0, 0);
  Tauola::initialize();
  const double m_tau = parmas_.amtau;

  out << "# iev pdg px py pz E\n";
  out.precision(9);
  out.setf(std::ios::scientific);

  std::map<std::string, long> modes;   // "mu", "e", "h" (hadronic) -> count
  long n_tau = 0, n_bad = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    long iev; int pdg; double px, py, pz, E, hel;
    if (!(ss >> iev >> pdg >> px >> py >> pz >> E >> hel)) continue;

    TauolaHEPEVTEvent evt;
    TauolaHEPEVTParticle *tau =
        new TauolaHEPEVTParticle(pdg, 1, px, py, pz, E, m_tau, -1, -1, -1, -1);
    evt.addParticle(tau);
    const double polz = pdg > 0 ? -hel : hel;   // see the header comment
    Tauola::decayOne(tau, false, 0.0, 0.0, polz);

    double sum[4] = {0, 0, 0, 0};
    bool has_mu = false, has_e = false;
    for (int i = 1; i < evt.getParticleCount(); ++i) {
      TauolaHEPEVTParticle *d = evt.getParticle(i);
      if (d->getStatus() != 1) continue;
      out << iev << ' ' << d->getPdgID() << ' ' << d->getPx() << ' ' << d->getPy()
          << ' ' << d->getPz() << ' ' << d->getE() << '\n';
      sum[0] += d->getPx(); sum[1] += d->getPy(); sum[2] += d->getPz(); sum[3] += d->getE();
      if (std::abs(d->getPdgID()) == 13) has_mu = true;
      if (std::abs(d->getPdgID()) == 11) has_e = true;
    }
    ++n_tau;
    ++modes[has_mu ? "mu" : has_e ? "e" : "h"];
    if (std::fabs(sum[3] - E) > 1e-3 * E || std::fabs(sum[2] - pz) > 1e-3 * E) ++n_bad;
  }

  std::cerr << n_tau << " taus decayed (seed " << seed << "): "
            << "mu " << 100.0 * modes["mu"] / n_tau << "%, "
            << "e " << 100.0 * modes["e"] / n_tau << "%, "
            << "hadronic " << 100.0 * modes["h"] / n_tau << "%; "
            << n_bad << " with energy-momentum mismatch > 0.1%\n";
  return 0;
}
