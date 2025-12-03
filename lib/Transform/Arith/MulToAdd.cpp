#include "MulToAdd.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdint>
#include <iostream>

namespace mlir {
namespace tutorial {

#define GEN_PASS_DEF_MULTOADD
#include "ArithPasses.h.inc"

using arith::AddIOp;
using arith::ConstantOp;
using arith::MulIOp;

// Replace y = C*x with y = C/2*x + C/2*x, when C is a power of 2, otherwise do
// nothing.
struct PowerOfTwoExpand : public OpRewritePattern<MulIOp> {
  PowerOfTwoExpand(mlir::MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!rhsDefiningOp) {
      return failure();
    }

    auto value = rhsDefiningOp.value();
    if ((value & (value - 1))) {
      return failure();
    }

    auto newConst = rewriter.create<arith::ShRSIOp>(
        rhs.getLoc(), rhs,
        rewriter.create<ConstantOp>(rhsDefiningOp->getLoc(),
                                    rewriter.getIntegerAttr(rhs.getType(), 1)));
    auto newMul = rewriter.create<arith::MulIOp>(op->getLoc(), lhs, newConst);
    auto newAdd = rewriter.create<AddIOp>(op->getLoc(), newMul, newMul);
    rewriter.replaceOp(op, {newAdd});
    rewriter.eraseOp(rhsDefiningOp);

    return success();
  }
};

// Replace y = 9*x with y = 8*x + x
struct PeelFromMul : public OpRewritePattern<MulIOp> {
  PeelFromMul(mlir::MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!rhsDefiningOp) {
      return failure();
    }

    auto value = rhsDefiningOp.value();
    auto newConst = rewriter.create<ConstantOp>(
        rhsDefiningOp->getLoc(),
        rewriter.getIntegerAttr(rhs.getType(), value - 1));
    auto newMul = rewriter.create<arith::MulIOp>(op->getLoc(), lhs, newConst);
    auto newAdd = rewriter.create<AddIOp>(op->getLoc(), newMul, lhs);
    rewriter.replaceOp(op, {newAdd});
    rewriter.eraseOp(rhsDefiningOp);

    return success();
  }
};

struct MulToAdd : impl::MulToAddBase<MulToAdd> {
  using MulToAddBase::MulToAddBase;

  void runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<PowerOfTwoExpand>(&getContext());
    patterns.add<PeelFromMul>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace tutorial
} // namespace mlir