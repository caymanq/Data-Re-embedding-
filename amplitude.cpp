#include <bits/stdc++.h>
using namespace std;

// --------- complex helpers ---------
using cd = complex<double>;
const double PI = 3.14159265358979323846;

// --------- state utilities ----------
inline size_t bit(size_t i, int q) { return (i >> q) & 1ULL; }

// Apply a 2x2 single-qubit unitary U to target qubit q (little-endian: qubit 0 = LSB)
void apply_single_qubit(vector<cd>& psi, int q, const array<cd,4>& U) {
    const size_t N = psi.size(); // 2^n
    const size_t mask = (1ULL << q);
    for (size_t i = 0; i < N; ++i) {
        if ((i & mask) == 0) {
            size_t i0 = i;
            size_t i1 = i | mask;
            cd a = psi[i0], b = psi[i1];
            // [u00 u01; u10 u11] * [a; b]
            psi[i0] = U[0]*a + U[1]*b;
            psi[i1] = U[2]*a + U[3]*b;
        }
    }
}

// Controlled-Z on qubits (a as control, b as target) — phase -1 on |11>
void apply_CZ(vector<cd>& psi, int a, int b) {
    if (a == b) return;
    const size_t N = psi.size();
    for (size_t i = 0; i < N; ++i) {
        if (bit(i,a) && bit(i,b)) psi[i] = -psi[i];
    }
}

// --------- rotations ----------
array<cd,4> RY(double theta) {
    double c = cos(theta/2.0);
    double s = sin(theta/2.0);
    // [[c, -s],[s, c]]
    return { cd(c,0), cd(-s,0), cd(s,0), cd(c,0) };
}
array<cd,4> RZ(double theta) {
    // diag(e^{-iθ/2}, e^{+iθ/2})
    cd e0 = exp(cd(0, -theta/2.0));
    cd e1 = exp(cd(0,  theta/2.0));
    return { e0, cd(0,0), cd(0,0), e1 };
}

// --------- amplitude embedding via Householder unitary ----------
// Given a real, normalized |x> (length D) we build a Householder reflector
// U = I - 2 |u><u| with |u> ∝ |e0> - |x>, so that U |e0> = |x>.
// We NEVER form full matrices; we store only |u>, and apply Uψ = ψ - 2<u|ψ>|u>.
struct Householder {
    vector<double> u; // real |u> (normalize to unit length)
    bool identity = false; // true if |x> == |e0>

    // Build from normalized real x
    explicit Householder(const vector<double>& x_norm) {
        size_t D = x_norm.size();
        u.assign(D, 0.0);
        // e0 = [1,0,0,...]; w = e0 - x
        u[0] = 1.0 - x_norm[0];
        for (size_t k = 1; k < D; ++k) u[k] = -x_norm[k];

        // if ||w|| == 0 => x == e0 => U = I
        double nn = 0.0; for (double v : u) nn += v*v;
        if (nn < 1e-16) { identity = true; return; }
        double inv = 1.0 / sqrt(nn);
        for (double &v : u) v *= inv; // u normalized
    }

    // Apply U to complex state psi (Uψ = ψ - 2<u|ψ>|u>)
    void apply(vector<cd>& psi) const {
        if (identity) return;
        // alpha = <u|psi> (u real)
        cd alpha = 0.0;
        for (size_t k = 0; k < u.size(); ++k) alpha += conj(cd(u[k],0)) * psi[k];
        cd factor = 2.0 * alpha;
        for (size_t k = 0; k < u.size(); ++k) psi[k] -= factor * cd(u[k],0);
    }
};

// L2 normalize real vector (in-place); returns norm
double l2_normalize(vector<double>& x) {
    double s = 0.0; for (double v : x) s += v*v;
    double n = sqrt(max(s, 1e-30));
    for (double& v : x) v /= n;
    return n;
}

// --------- variational block V(ω): per-qubit RY,RZ + ring CZ ----------
struct VBlock {
    int n;
    // params: for each qubit q, two angles {thetay, thetaz}
    vector<double> thetay, thetaz;
    explicit VBlock(int n_qubits): n(n_qubits), thetay(n), thetaz(n) {}

    void set_params(const vector<double>& wy, const vector<double>& wz) {
        thetay = wy; thetaz = wz;
    }

    void apply(vector<cd>& psi) const {
        for (int q = 0; q < n; ++q) apply_single_qubit(psi, q, RY(thetay[q]));
        for (int q = 0; q < n; ++q) apply_single_qubit(psi, q, RZ(thetaz[q]));
        // ring entanglement via CZ
        for (int q = 0; q < n; ++q) apply_CZ(psi, q, (q+1)%n);
    }
};

// --------- observables: expectation of tensor products of Z ----------
double expval_Z_mask(const vector<cd>& psi, unsigned maskZ) {
    // maskZ bit = 1 means include Z on that qubit
    // <Z...Z> = sum_i s(i)*|psi_i|^2, where s(i)=(-1)^{popcount(i & maskZ)}
    double acc = 0.0;
    for (size_t i = 0; i < psi.size(); ++i) {
        int parity = __builtin_popcountll(i & maskZ) & 1;
        double sign = parity ? -1.0 : 1.0;
        acc += sign * norm(psi[i]); // |psi_i|^2
    }
    return acc;
}

// Convenience masks for n=2: ZI, IZ, ZZ
double expval_ZI(const vector<cd>& psi) { return expval_Z_mask(psi, /*mask*/1u<<0); }
double expval_IZ(const vector<cd>& psi) { return expval_Z_mask(psi, /*mask*/1u<<1); }
double expval_ZZ(const vector<cd>& psi) { return expval_Z_mask(psi, (1u<<0)|(1u<<1)); }

// --------- QNN forward: amplitude reuploading ----------
struct QNN {
    int n;          // number of qubits
    int L;          // reuploading depth
    vector<VBlock> layers;

    explicit QNN(int n_qubits, int L_layers): n(n_qubits), L(L_layers) {
        for (int i = 0; i < L; ++i) layers.emplace_back(n);
    }

    // Set all variational params: wy[L][n], wz[L][n]
    void set_params(const vector<vector<double>>& wy, const vector<vector<double>>& wz) {
        for (int i = 0; i < L; ++i) layers[i].set_params(wy[i], wz[i]);
    }

    // Forward pass: given real x (length 2^n), return logits [ZI, IZ, ZZ]
    array<double,3> forward(const vector<double>& x_raw) const {
        const size_t D = size_t(1) << n;
        if (x_raw.size() != D) throw runtime_error("x dimension must be 2^n");

        vector<double> x = x_raw;
        l2_normalize(x);
        Householder Ux(x); // unitary for amplitude embedding

        // start in |0...0>
        vector<cd> psi(D, cd(0.0,0.0));
        psi[0] = cd(1.0,0.0);

        for (int i = 0; i < L; ++i) {
            Ux.apply(psi);          // U_φ(x)
            layers[i].apply(psi);   // V(ω_i)
        }

        return { expval_ZI(psi), expval_IZ(psi), expval_ZZ(psi) };
    }
};

// -------------- demo main --------------
int main() {
    ios::sync_with_stdio(false);

    // Two-qubit demo (fits 4 Iris features by amplitude)
    const int n = 2;
    const int L = 3; // reuploading depth
    QNN qnn(n, L);

    // Random small params for demo (you would train these)
    mt19937_64 rng(42);
    normal_distribution<double> N01(0.0, 1.0);
    vector<vector<double>> wy(L, vector<double>(n)), wz(L, vector<double>(n));
    for (int i = 0; i < L; ++i)
        for (int q = 0; q < n; ++q) {
            wy[i][q] = 0.3 * N01(rng);
            wz[i][q] = 0.3 * N01(rng);
        }
    qnn.set_params(wy, wz);

    // Example 4D input (e.g., one standardized Iris row)
    vector<double> x = { 0.2, -1.1, 0.6, 0.1 }; // length 4 = 2^2

    auto logits = qnn.forward(x);
    cout << fixed << setprecision(6);
    cout << "logits ( <ZI>, <IZ>, <ZZ> ) = "
         << logits[0] << ", " << logits[1] << ", " << logits[2] << "\n";

    // If you want probabilities for 3-class: apply a softmax (optional temperature)
    auto softmax = [](array<double,3> z, double T=1.5) {
        double zmax = max(z[0], max(z[1], z[2]));
        array<double,3> e;
        double Z = 0.0;
        for (int k=0;k<3;++k) { e[k] = exp((z[k]-zmax)/T); Z += e[k]; }
        for (int k=0;k<3;++k) e[k] /= Z;
        return e;
    };
    auto probs = softmax(logits);
    cout << "probs = " << probs[0] << ", " << probs[1] << ", " << probs[2] << "\n";

    return 0;
}
