#ifndef OTAGRUM_CORRECTEDMUTUALINFORMATION_HXX
#define OTAGRUM_CORRECTEDMUTUALINFORMATION_HXX

#include <agrum/base/core/hashTable.h>

#include <utility>
#include <vector>

#include <openturns/Point.hxx>
#include <openturns/Sample.hxx>
#include <openturns/NormalCopula.hxx>

#include "otagrum/IndicesManip.hxx"

namespace OTAGRUM
{

class OTAGRUM_API CorrectedMutualInformation : public OT::Object
{
public:
  // Constructor
  explicit CorrectedMutualInformation(const OT::Sample &data);

  void setAlpha(const double alpha);
  double getAlpha() const;

  OT::UnsignedInteger getSize() const;
  OT::UnsignedInteger getDimension() const;
  OT::Description getDataDescription() const;

  void clearCaches() const;

  double compute2PtCorrectedInformation(const OT::UnsignedInteger X,
                                        const OT::UnsignedInteger Y,
                                        const OT::Indices &U = OT::Indices());
  double compute2PtCorrectedInformation(const OT::Indices &X, const OT::Indices &Y);
  double compute3PtCorrectedInformation(const OT::UnsignedInteger X,
                                        const OT::UnsignedInteger Y,
                                        const OT::UnsignedInteger Z,
                                        const OT::Indices &U = OT::Indices());

  // NoCorr   : no penalty, the raw plug-in information is used.
  // Naive    : constant penalty alpha, the historical CMIIC behaviour.
  // Analytic : sample split, construction on the first half, evaluation on the
  //            second half, and the analytic threshold mu_n + z_{1-alpha} sigma_tot
  //            with sigma_tot^2 = sigma_n^2 + s^2 / n_eval. Here alpha is the
  //            level of the test. Bernstein copulas only.
  enum class KModeTypes {NoCorr, Naive, Analytic};
  enum class CModeTypes {Gaussian, Bernstein};

  void setKMode(CorrectedMutualInformation::KModeTypes kmode);
  void setCMode(CorrectedMutualInformation::CModeTypes cmode);

  KModeTypes getKMode();
  CModeTypes getCMode();

private:
  void clearHCache() const;

  std::string GetKey(OT::Indices l) const;
  std::string GetKey(OT::Indices l, const OT::UnsignedInteger k) const;
  OT::UnsignedInteger GetK(const OT::UnsignedInteger size,
                           const OT::UnsignedInteger dimension);


  double computeCrossEntropy(const OT::Indices &variables, const int K);

  // Analytic mode. Size used to choose K, the construction half.
  OT::UnsignedInteger sizeForK() const;
  // Log-density of the Bernstein copula of `variables`, built on the
  // construction half, at the evaluation points. Cached by (variables, K).
  // A block of dimension <= 1 gives the null vector.
  OT::Point computeLogPDFOnEval(const OT::Indices &variables, const int K);
  // a = K sum_r (int_0^1 P(Bin(K-1,u)=r)^2 du) and
  // S = sum_{r,s} (int_0^1 P(Bin(K-1,u)=r) P(Bin(K-1,u)=s) du)^2, closed form.
  static void BinomialConstants(const OT::UnsignedInteger K, double &a, double &S);
  // Corrected information of a signed sum of blocks: mean of the pointwise
  // log-ratio on the evaluation half minus mu_n + z_{1-alpha} sigma_tot.
  // blocks: (variables, sign), the first block being the largest one.
  double computeAnalyticCorrectedInformation(
    const std::vector< std::pair< OT::Indices, int > > &blocks);

  double compute2PtInformation(const OT::UnsignedInteger X,
                               const OT::UnsignedInteger Y,
                               const OT::Indices &U = OT::Indices());
  double compute2PtInformation(const OT::Indices &X, const OT::Indices &Y);

  double compute2PtPenalty(
    // Commented until a better correction is found
    //const OT::UnsignedInteger X,
    //const OT::UnsignedInteger Y,
    //const OT::Indices &U = OT::Indices()
  );

  double compute3PtInformation(const OT::UnsignedInteger X,
                               const OT::UnsignedInteger Y,
                               const OT::UnsignedInteger Z,
                               const OT::Indices &U = OT::Indices());
  double compute3PtPenalty(
    // Commented until a better correction is found
    //const OT::UnsignedInteger X,
    //const OT::UnsignedInteger Y,
    //const OT::UnsignedInteger Z,
    //const OT::Indices &U = OT::Indices()
  );

  mutable gum::HashTable< std::string, double > HCache_;
  mutable gum::HashTable< std::string, OT::Point > logPDFCache_;
  OT::Sample data_;
  OT::Sample fit_;   // first half of data_, Analytic mode
  OT::Sample eval_;  // second half of data_, Analytic mode
  KModeTypes kmode_{KModeTypes::Naive};
  CModeTypes cmode_{CModeTypes::Bernstein};
  double alpha_ = 0.01;
};

}

#endif // OTAGRUM_CORRECTEDMUTUALINFORMATION_HXX
